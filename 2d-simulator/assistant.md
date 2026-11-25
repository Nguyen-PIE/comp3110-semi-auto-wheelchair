# The learned assistant (first RL step)

The assistant drives the chair to its goal, predicts where nearby people are walking, and slows
down or steers around them. It works in every scenario: the generated rooms (corridor, ward,
concourse) and the floor plans (hospital, office, warehouse, or your own). The user can take
control at any time with **O**. When they hand control back, the chair carries on to the same
goal.

It is the first reinforcement learning part of the project. The policy does not drive the chair
directly. It adjusts a route follower, which keeps it away from walls and lets it train in under
an hour on a laptop CPU. When nobody is in the way, the policy is not used at all and the chair
simply follows its route. A simple safety stop sits underneath the policy for people right in
front of the chair.

## The scenario it is built for

| Step in the scenario | What does it in the simulator |
| --- | --- |
| The user picks a destination | The scenario's goal. Choosing a named room came later, in 3D ([webots_open_world.md](../docs/webots_open_world.md)). |
| The chair routes there automatically | The route follower (A* on the floor plan or the room's plan), `baselines/route_follower.py` |
| Sensors feed the model | People tracker and wall sensor, `assist/tracker.py` and `assist/sensors.py` |
| It estimates a person's speed and a collision | The tracker's prediction, counted in the line under the map in the demo |
| It slows down and turns slightly to go around | The policy's adjustment, `assist/controller.py` |
| The user takes the joystick, then hands back | **O** in the demo, then the chair carries on along its route from where it is |

## How one step works

Every 0.1 s:

1. **Route.** The route follower plans a route to the goal once, around walls and furniture,
   keeping to the middle of corridors and doorways. It plans again only when something changes:
   someone starts or stops standing still near the chair (they are planned around), or the chair
   ends up more than 0.75 m off its route (after the user drove, for example). It never turns
   back for a waypoint it has already passed. Floor plans come from the map. In the generated
   rooms a plan of the room's walls, tables, chairs and plants is built at each reset.
2. **Driving along the route.** The chair aims at a point 1 m further along the route and drives
   on the arc towards it, slowing down only as much as its top turn rate (0.8 rad/s) needs. It
   turns on the spot only when the route doubles back. In tight spaces it slows down and follows
   the route more closely. (Before, it stopped and turned on the spot at every bend, which looked
   like it was hesitating.)
3. **People tracker.** It gets the positions of people within 8 m, with 3 cm of noise as a lidar
   or camera tracker would give. From the last second of positions it estimates each person's
   velocity, then predicts when that person will come closest to the chair and how close
   (looking up to 4 s ahead). A predicted gap under 0.3 m counts as a predicted collision, and it
   stays one until the predicted gap grows past 0.5 m, so noise does not switch it on and off.
   It never reads the simulator's true velocities, so the same code can run on real sensor data.
4. **Wall sensor.** The distance to the nearest wall or piece of furniture in 16 directions, up
   to 4 m.
5. **Does anyone matter?** The policy is used only while someone is predicted to come too close,
   or someone is on the next 3 m of the route (and not walking away along it), and for 1 s after
   that. Otherwise the chair follows its route at full speed. This is why it no longer slows down
   or looks around when the way ahead is clear.
6. **Policy.** A small neural network (see [Training](#training)) reads 55 numbers:
   - 5 for the route and the chair's motion
   - its last 2 outputs
   - the 16 wall distances
   - 8 for each of the 4 nearest people: position, velocity relative to the chair, current gap,
     predicted closest gap and time to it, and whether the slot is used.

   It outputs 2 numbers: a speed factor (0 to 100% of normal) and a steering offset (up to 34
   degrees left or right of the route).
7. **Controller.** Like a real powered chair, it brakes at once but takes about a second to
   reach full speed. The steering offset can change by up to 1 rad/s when swerving, and returns
   to the route more gently (0.4 rad/s). It is cut back when it would steer the chair towards a
   wall or piece of furniture less than 0.5 m from its side.
8. **Safety stop.** It overrides the policy. The chair does not drive forward with someone closer
   than 0.4 m straight ahead (less for someone off to the side), or with someone brushing past its
   side, though it can still turn. People beside or behind it do not stop it. This is the first
   piece of the planned safety layer. It does not apply while the user is driving.
9. **Way out of a gap.** If the safety stop has held the chair still for 2 s (boxed in between
   people, for example), it looks all round for open space using the wall sensor and the people
   it tracks. Of the directions with at least 0.7 m clear, it takes the most open one, nearest the
   route to the goal, turns to it and drives out at 35% speed for 1 m (at most 6 s, giving up if
   the space ahead drops below 0.3 m, and does nothing if boxed in all round). The route follower
   then re-plans from where the chair is. It only knows what the sensors see, so someone hidden
   behind another person is not avoided until they come into view.

## Training

```bash
python training/train_assist.py                                        # about 45 minutes on 2 cores
python training/train_assist.py --demo-steps 5000 --timesteps 20000    # a quick try
```

Training has two stages:

1. **Head start (behaviour cloning).** A cautious hand-written rule, `training/teacher.py`,
   drives for 40,000 decisions. It stops for anyone close in front of the chair or brushing past
   its side, slows and steers away when the tracker predicts a collision, and steers around
   people standing in its path. It picks a side to pass each person on once and keeps to it,
   taking the side with more room and keeping left when both have room. The policy network
   learns to copy it. Starting from scratch instead, PPO spent its first 100,000 steps crashing
   into people and barely improved.
2. **Reinforcement learning (PPO).** The policy tries small variations on what it learned and
   keeps the ones that earn more reward.

Each step of training is one decision: after the policy acts, the chair follows its route on its
own until someone matters again, and the reward adds up in between. Each episode is one trip in
one of the eight scenarios, floor plans twice as often as the rooms. Scene seeds come from the
training range (0 to 9,999). Progress is printed every 10,000 steps. The model is saved to
`models/assist_ppo.zip`, and snapshots go to `models/checkpoints/` every 25,000 steps.

`--dagger-rounds 2` adds correction rounds between the two stages: the copy drives while the rule
labels what it should have done, and the network trains on those states too (DAgger). It is off
by default.

The reward, in `training/assist_env.py`:

| Event | Reward |
| --- | --- |
| Each metre moved along the route | +1 |
| Reaching the goal | +10 |
| Any collision (person, wall or furniture) | -30, so the policy never trades a collision for speed |
| Each step driving closer than 0.5 m to a person | up to -0.25, scaled by the chair's speed |
| Each step | -0.005, so waiting forever does not pay |
| Changing the adjustment abruptly, or swinging the chair's turn rate from one step to the next | a penalty, for smooth driving without weaving |

`models/assist_ppo.zip` is kept in Git (about 600 KB), so teammates can run the demo without
training. Retrain after changing the observation, the reward, the controller or the people's
behaviour.

## Results

Validation scenes (seeds 10,000 to 10,019, never used in training): 20 trips in each of the eight
scenarios, 160 in total. Time limits: 60 s in the rooms, 90 s on the floor plans.

| Driver | Reached the goal | Hit a person | Ran out of time | Average trip time |
| --- | --- | --- | --- | --- |
| Route follower, ignores people (`--driver route_follower`) | 96 (60%) | 64 (40%) | 0 | |
| Cautious rule it learned from (`--driver rule`) | 160 (100%) | 0 | 0 | 19.3 s |
| **Learned assistant** (`--driver assist`, the model in Git) | **160 (100%)** | **0** | **0** | **17.0 s** |

The learned assistant also reached the goal in all 240 further validation trips we tried (seeds
10,020 to 10,049 in every scenario), again with no collisions and no timeouts: 400 out of 400 in
total. Its longest trip took 44 s. With the chair standing still, nobody walks into it (0 of 30
trips of 45 s).

Per scenario (learned assistant, first 160 trips):

| Scenario | Reached the goal | Average trip | Longest trip |
| --- | --- | --- | --- |
| `s1_corridor` | 20/20 | 18.2 s | 23.9 s |
| `s2_ward` | 20/20 | 14.1 s | 22.5 s |
| `s3_concourse_low` | 20/20 | 12.3 s | 18.5 s |
| `s3_concourse_mid` | 20/20 | 13.2 s | 18.6 s |
| `s3_concourse_high` | 20/20 | 14.2 s | 24.1 s |
| `fp_hospital` | 20/20 | 24.2 s | 38.2 s |
| `fp_office` | 20/20 | 23.0 s | 40.2 s |
| `fp_warehouse` | 20/20 | 16.8 s | 25.1 s |

**Smoother and quicker than the previous version.** On the same 10 scenes in each of the
hospital, office, warehouse and corridor, compared with the previous version (which stopped to
turn at every bend, re-planned every 3 s, used the policy all the time, and ran with the old
pedestrians):

| | Previous version | Now |
| --- | --- | --- |
| Hospital, office, warehouse: average trip | 57 s, 49 s, 25 s | 27 s, 25 s, 16 s |
| Corridor | 9 of 10 reached the goal, 39 s on average | 10 of 10, 18 s |
| Stopping and starting again with nobody near (per minute) | 2.6 to 6.2 | 0 |
| Standing still with nobody near (seconds per minute) | 6 to 9.5 | 0.5 to 2 (turning towards the route at the start) |
| Weaving: turning left, right and left again within 1.5 s (per minute) | 3.6 to 7.8 | 1.3 to 4.5 |
| People giving up on their destination before reaching it (per person per minute) | 0.3 to 1.2 | 0 |

What this shows:

- Predicting where people are walking, then slowing down or steering away, takes collisions with
  people from 40% of trips to none, in rooms and on floor plans alike.
- Reinforcement learning made the policy about 12% quicker than the rule it copied, with the same
  safety record. A penalty on swinging the turn rate keeps it from weaving more than the rule.
- Leaving the policy out when nobody is in the way, and turning while moving, removed the
  hesitations: the chair no longer stops or looks around when the way ahead is clear.

How the model in Git was chosen: the snapshots after 50,000 and 100,000 RL decisions were checked
on the first 10 validation scenes of each scenario. Both reached the goal every time, and the
100,000-decision one was quicker and weaved less. (A run without the penalty on swinging the
turn rate was as quick but weaved more, up to 6 times a minute.) The test seeds (20,000 onwards)
are kept for the final evaluation in Week 10.

To reproduce:

```bash
python eval/run_episodes.py --scenario all --episodes 20 --driver assist
```

Add `--save-paths` to keep the chair's position at every step (`data/paths/`), to compare its
paths with another driver or a later version. In the demo, the dotted orange line shows the
path driven so far next to the shortest route in green.

## How people walk

SocNavGym's own people made the scenes unrealistic: they froze in front of the chair, stopped dead
whenever they turned, and changed destination for no reason (after 50 s of real time, so it even
depended on how fast the computer was). Our watchdog then gave anyone stuck for 3 s a new
destination, so people froze and then suddenly walked off somewhere else. `envs/pedestrians.py`
now decides how people walk, in every scenario:

- **A destination they keep.** Each walking person has a destination and walks a route to it
  (A* on the floor plan or the room's plan), planned around the chair. They get a new destination
  only when they arrive. If the chair is parked on their destination, being next to it counts as
  arriving.
- **Passing the chair.** A person who sees the chair on their path (allowing for its motion)
  chooses a side once and walks past with room to spare, the way people pass a wheelchair.
  Attentive people (70%, `human.prob_to_avoid_robot`) react about 3 s ahead and leave about
  0.5 m. Distracted people react only about 1 s ahead and leave about 0.15 m, which gives the
  assistant the hard cases it must handle.
- **No room to pass.** They step to the side and wait for the chair, then carry on to the same
  destination. If the chair is waiting too, they take another way round it.
- **Smooth walking.** Speed and direction change gradually (up to 2 m/s²), nobody stops just to
  turn, and they slow down as they arrive.
- **Their own pace.** Each person's walking speed is drawn from a normal distribution (mean
  1.15 m/s, standard deviation 0.2, between 0.6 and 1.6), so a crowd has strollers and people in
  a hurry.
- **On the phone.** 20% of walkers are slow (0.5 to 0.85 m/s), never watch for the chair, drift
  from side to side, and stop now and then to read.
- **Stopping.** 30% of the other walkers stop for 2 to 8 s every so often. Everyone has a 40%
  chance of waiting 2 to 10 s on arriving before choosing the next destination. A stopped person
  still steps aside for a chair coming straight at them, if they are attentive.
- **Groups.** 30% of walkers go in pairs or threes (30% of the groups are threes). Members walk
  side by side at the leader's pace, wait when the leader stops, and the leader slows down if the
  others fall behind.
- **Getting stuck.** Someone who has not moved for 3 s (for example two people blocking each
  other) plans a new route to the same destination.
- **Never into the chair.** A step that would end inside the chair is undone, so every collision
  involves the chair moving towards someone.

Scenes where the chair could not reach its goal at all (boxed in by furniture, or by people who
stand still all episode) are drawn again with a seed derived from the original one, so each seed
still always gives the same scene.

Who is on the phone, who stops and who walks in a group is drawn once per scene from its seed
(`envs/behaviours.py`, used by both the 2D simulator and Webots). The numbers are in
`project.behaviour` in `envs/configs/base.yaml`, and any of them can be overridden, for example
`make_env("s3_concourse_mid", overrides={"project": {"behaviour": {"phone_share": 0.5}}})`. Setting
`phone_share`, `group_share`, `pause_share` and `dwell_chance` to 0 gives plain walkers.

To fit the speed, stopping and group numbers to real pedestrian data, run
`python scripts/fit_behaviour.py <file>` (see the script's help for the file layouts). It prints
a block to paste into `base.yaml`. Phone use cannot be read from positions, so it stays a setting.
Indoor datasets such as THÖR, ATC or Grand Central suit a hospital or office better than the
outdoor ETH and UCY sets.

Results from before these changes cannot be compared with results after them (people also walk
faster now: `human.max_advance_human` is 1.6 m/s).

How comfortable the driving is (stops and starts, braking, speed near people, how well the tracker
predicted) is measured separately from success and collisions: see [evaluation.md](../docs/evaluation.md).

## Limitations

- Destinations are the scenario's goal. Choosing a named room needs a structured building plan, which
  the 3D simulator adds ([webots_open_world.md](../docs/webots_open_world.md)).
- The policy sees only the 4 nearest people. Your own floor plans work, but check them with
  `eval/run_episodes.py` first.
- The user's take-overs come from the keyboard. A simulated user was not built.
- RL gives a modest gain over the copied rule, and most of the safety comes from the
  predictions, the rule it copies and the safety stop. Learning from the user's take-overs was not
  built.
- The tracker and wall sensor read positions from the simulator, with noise. The 3D simulator adds a
  lidar and a camera ([webots/README.md](../webots/README.md)).

## References

- **Behaviour cloning.** Pomerleau, D. A. *Efficient Training of Artificial Neural Networks for Autonomous Navigation.* Neural Computation 3(1), 88-97, 1991.
- **PPO.** Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- **Stable-Baselines3.** Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268), 2021. [Code](https://github.com/DLR-RM/stable-baselines3).
- **A\*.** Hart, P. E., Nilsson, N. J. and Raphael, B. *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.* IEEE Transactions on Systems Science and Cybernetics 4(2), 100-107, 1968.
- **SocNavGym**, the simulator: [socnavgym.md](socnavgym.md#references).
