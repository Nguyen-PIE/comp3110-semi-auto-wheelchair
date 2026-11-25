# Working with SocNavGym

A guide for writing code against the simulator: what goes in and out of the environment, how to
change scenarios, and the pitfalls to know before training. For installing and running the demo,
see the [README](README.md).

## The basics

SocNavGym simulates a robot (here, the wheelchair) moving among people, tables, chairs and plants,
as a standard [Gymnasium](https://gymnasium.farama.org) environment. We pin version 0.2.0
(`SocNavGym-v2`).

Always create environments with `envs.make_env.make_env`, never `gym.make` directly. It merges our
scenario configs, applies our fixes to SocNavGym (see [below](#fixes-we-apply)), and uses the
floor-plan version when needed. The README shows the basic `reset` and `step` loop.

`env.unwrapped` is the SocNavGym object itself: `env.unwrapped.robot` is the chair (`x`, `y`,
`orientation`, `goal_x`, `goal_y`), and `env.unwrapped.dynamic_humans` and `static_humans` are the
people (`x`, `y`, `orientation`, `speed`). On floor plans, `env.unwrapped.plan` is the map, with
`plan.route(start, goal)` and `plan.clearance_at(x, y)`. In the generated rooms the same kind of
plan, built from the room's walls and furniture at each reset, is `env.unwrapped._room_plan`
(`baselines.route_follower.plan_of(env)` returns whichever applies).

## Action

Three numbers, each from -1 to 1:

| Index | Meaning | Scaled to |
| --- | --- | --- |
| 0 | Forward speed | 0 to 1.2 m/s (the chair cannot reverse, so negative values stop it) |
| 1 | Sideways speed | Ignored, because the chair is differential drive |
| 2 | Turn rate | -0.8 to 0.8 rad/s, positive turns left |

## Observation

A dictionary of flat arrays, all in the chair's own frame (x forward, y left):

| Key | Contents |
| --- | --- |
| `robot` | 7 numbers: goal x, goal y, goal radius, sin and cos of the goal heading, heading tolerance, chair radius |
| `humans` | 8 numbers per person, padded with zeros to the most people the scenario can have |
| `tables`, `chairs`, `plants`, `laptops` | 8 numbers per object (empty if the scenario has none) |
| `walls` | 8 numbers per 1 m wall segment in rooms, all zeros on floor plans |

The 8 numbers per person or object are: x, y, sin and cos of its heading, radius, speed relative to
the chair, turning rate, and (people only) 1 if the chair is in the person's view.

Two things to design around:

- **The order of people is shuffled every step**, so a policy cannot rely on a person keeping the
  same slot.
- **Floor-plan walls are not in the observation.** The assistant uses its own wall sensor
  (`assist/sensors.py`, laser-style distances from `plan.clearance_at`) and builds its own
  observation (`assist/controller.py`).

## Reward and info

The reward is SocNavGym's built-in DSRNN reward:

| Event | Reward |
| --- | --- |
| Reaching the goal | +10 |
| Colliding, or leaving the map | -10 |
| Running out of time | -5 |
| Any other step | +0.1 per metre moved towards the goal, minus a small penalty within 0.6 m of a person |

Progress is measured in a straight line to the goal, which on floor plans can point through
walls.

The most useful `info` fields:

| Key | Meaning |
| --- | --- |
| `SUCCESS`, `TIMEOUT`, `OUT_OF_MAP` | How the episode ended |
| `COLLISION` (and `COLLISION_HUMAN`, `_OBJECT`, `_WALL`) | What the chair hit |
| `MINIMUM_DISTANCE_TO_HUMAN` | Centre-to-centre distance to the closest person in metres. Subtract 0.76 for the gap between edges |
| `PATH_LENGTH`, `TIME_TO_REACH_GOAL` | Metres driven, and seconds taken, set on success |
| `JERK_AVG` | Smoothness (lower is smoother) |
| `DISTANCE_TO_WALL` | Floor plans only: gap between the chair and the nearest wall |

`eval/metrics.py` turns these into the project's metrics.

## Changing scenarios

Each scenario is `envs/configs/base.yaml` plus a small file with only the values it changes
(floor-plan scenarios also add `floorplan.yaml`). The settings you are most likely to change:

| Setting | Meaning |
| --- | --- |
| `env.min_dynamic_humans`, `max_dynamic_humans` | People walking (drawn at random in this range) |
| `env.min_static_humans`, `max_static_humans` | People standing alone |
| `env.min_h_h_static_interactions_non_dispersing` (and `max_`) | Groups standing and talking |
| `env.min_tables`, `min_chairs`, `min_plants` (and `max_`) | Furniture |
| `env.set_shape`, `min_map_x` to `max_map_y` | Room shape (`"square"` or `"rectangle"`) and size in metres |
| `human.prob_to_avoid_robot` | Share of people who are attentive to the chair (0.7): they make way about 3 s ahead and leave about 0.5 m. The rest react only about 1 s ahead |
| `human.max_advance_human` | The fastest anyone walks, in m/s (1.6). Each person's own pace is in `project.behaviour` |
| `project.behaviour.*` | What walking people are like: speeds, share on the phone, stops, waits, groups (see [assistant.md](assistant.md#how-people-walk)) |
| `episode.episode_length` | Steps before a timeout (600 = 60 s) |
| `project.min_goal_distance` | Ours: minimum start-to-goal distance in metres |

To add a room, create `envs/configs/s4_lobby.yaml`:

```yaml
# S4 Lobby: a wide room with a bigger crowd and some plants.
env:
  set_shape: "rectangle"
  min_map_x: 16
  max_map_x: 18
  min_map_y: 10
  max_map_y: 12
  min_dynamic_humans: 8
  max_dynamic_humans: 12
  min_plants: 2
  max_plants: 4
project:
  min_goal_distance: 8
```

Then add `"s4_lobby": "S4 Lobby",` to `ROOM_SCENARIOS` in `envs/make_env.py`.

To change a setting in code without a new file, pass `overrides` in the same layout as the YAML
files, for example `make_env("fp_office", overrides={"env": {"max_dynamic_humans": 4}})`.

## Custom rewards

SocNavGym only loads its two built-in reward files, so a custom reward is a Gymnasium wrapper:

```python
import gymnasium as gym

from envs.make_env import make_env


class ProximityPenalty(gym.Wrapper):
    """Example: extra penalty while the chair is within 0.5 m of a person."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped
        gap = info["MINIMUM_DISTANCE_TO_HUMAN"] - u.ROBOT_RADIUS - u.HUMAN_DIAMETER / 2
        if gap < 0.5:
            reward -= 0.1
        return obs, reward, terminated, truncated, info


env = ProximityPenalty(make_env("s3_concourse_high"))
```

The assistant's reward is in `training/assist_env.py` (see [assistant.md](assistant.md)).

## Training with Stable-Baselines3

Remove the empty parts of the observation first: Stable-Baselines3's `predict()` fails on
zero-length arrays (tables, chairs, plants and laptops in most scenarios).

```python
from gymnasium.wrappers import FilterObservation
from stable_baselines3 import PPO

from envs.make_env import make_env

env = FilterObservation(make_env("s3_concourse_low"), filter_keys=["robot", "humans", "walls"])
model = PPO("MultiInputPolicy", env, verbose=1, seed=0)
model.learn(total_timesteps=1_000)       # a smoke test; real runs need 1,000,000 or more
model.save("models/ppo_s3_low")
```

Train on seeds 0 to 9,999 and check on 10,000 to 10,199. Trained models go in `models/`, which
Git ignores.

## Fixes we apply

SocNavGym's own README describes version 1, so trust the code and this guide instead. We work around
these problems without editing the library:

- **Start-up crash** from a typo (`WALL_SEGMENTSIZE`): patched in `envs/socnavgym_fixes.py`.
- **People walking at 6 cm/s:** its social force model assumes 1 s steps, and ours are 0.1 s. Rescaled
  in `envs/socnavgym_fixes.py` so people walk at about 1 m/s.
- **How people walk around the chair.** Its people froze in front of the chair (the pull towards
  their destination and the push away from the chair cancel out), stopped dead whenever they
  turned (a turn-rate limit meant for 1 s steps), swapped destinations after 50 s of real time,
  and walked through a parked chair. `envs/pedestrians.py` replaces this in every scenario: people
  walk a route to their destination and only get a new one when they arrive, pass the chair on a
  side they keep to, step aside and wait where there is no room, and change speed and direction
  smoothly. People also have characters (on the phone, stopping, walking in groups) from
  `envs/behaviours.py`. See [assistant.md](assistant.md#how-people-walk).
- **Steps into the chair** are undone (`envs/socnavgym_fixes.py`), so every collision involves the
  chair moving towards someone.
- **Arrivals not counted** about half the time, because a heading check does not wrap angles.
  `base.yaml` sets the heading tolerance above a full turn, so arriving is enough.
- **L-shaped rooms and corridors** are broken in v0.2, so we use rectangles.
- **Its example configs** use the old format and will not load, so use ours.

The library's code is in `third_party/SocNavGym/socnavgym/envs/` after setup: `socnavenv_v2.py`
(the environment), `utils/` (robot, people, objects) and `rewards/`.

## References

- **SocNavGym.** Kapoor, A., Swamy, S., Bachiller, P. and Manso, L. J. *SocNavGym: A Reinforcement Learning Gym for Social Navigation.* IEEE RO-MAN 2023. [Code](https://github.com/gnns4hri/SocNavGym).
- **Gymnasium.** Towers, M. et al. *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* arXiv:2407.17032, 2024. [gymnasium.farama.org](https://gymnasium.farama.org).
- **RVO2**, which SocNavGym needs. van den Berg, J., Guy, S. J., Lin, M. and Manocha, D. *Reciprocal n-Body Collision Avoidance.* Robotics Research (ISRR 2009), Springer, 2011. [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2).
- **Social force model.** Helbing, D. and Molnár, P. *Social Force Model for Pedestrian Dynamics.* Physical Review E 51(5), 4282-4286, 1995.
- **Stable-Baselines3.** Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268), 2021. [Code](https://github.com/DLR-RM/stable-baselines3).
- **PPO.** Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
