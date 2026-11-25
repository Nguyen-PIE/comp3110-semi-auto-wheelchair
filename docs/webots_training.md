# Fine-tuning the policy in Webots

How `train.py` fine-tunes the 2D policy in 3D, and what it gave. The short version is in
[webots/README.md](../webots/README.md#fine-tuning-in-3d). Running it on the AWS machines is in
[aws_training.md](aws_training.md).

## How it works

What fine-tuning does: `train.py` loads a copy of the 2D policy, lets it drive thousands of trips
in the training worlds and adjusts its weights with PPO, the method used in 2D (all weights can
change, and none are frozen). The result is saved as `models/assist_ppo_webots.zip`, and the 2D
policy file is never changed. Each run starts again from the 2D policy, unless `--start-from` names an
earlier 3D one.

`train.py` starts one Webots per environment (`--envs`), minimised and without drawing, on ports
1234, 1235 and so on, so close any Webots you have open first. It trains on the 28 training worlds in
rounds: each round every Webots gets the next world, and Webots restarts between rounds. Every
training trip also gets random variety: busier or quieter and more or less spread-out crowds, more
or fewer distracted people, a chair that responds a little more or less strongly or a step late,
lidar noise, older people, people stepping out of rooms and each of the building's events with a
30% chance (`Variety` in `bridge.py`, turned off by `--no-variety`). Half the trips have people near
the route, half spread out (`--spread-share`). Testing never uses this randomness.

## How to run

```bash
python webots/train.py --worlds corridor --envs 2 --timesteps 2000   # a quick check that it works
python webots/train.py --envs 14 --timesteps 300000                  # a real run, on the 16-core cloud machine
```

Run a real run inside `tmux`, so it keeps going if you disconnect.

## Results

All runs on Jayden's Windows PC, 20 trips per test world (2 November 2026):

| | 2D policy | Fine-tuned in 3D (50k steps) |
| --- | --- | --- |
| Spread: new furniture / new building | 100% / 97% | 100% / 100% |
| Busy: new furniture / new building | 92% / 95% | 92% / 100% |
| Events: new furniture / new building | 95% / 100% | 97% / 100% |
| Crashes (events) | 1, office | 1, depot |

- The differences are one to three trips out of 140: within luck.
- Nearly all failures are timeouts in busy crowds (hospital, office, depot): the chair waits for a
  gap that does not come.
- Both crashes happened with the chair standing still: a distracted walker (in the depot also a
  forklift) moved into it. That is why crashes are split into `hit_moving` and `hit_stopped`.
- Windows and Linux can give very slightly different physics, so a close pass can end as a touch
  on one and a near miss on the other.

A later fine-tune on the AWS machine, with the walkers trained on SiT and the proxemic reward, is the
latest model, `models/sit_prox.zip`. On the 540 validation trips with the SiT walkers it reached the goal
in 539 with no collisions and braked hard 9.6% less than the 2D policy
([evals.md](evals.md#retraining-with-the-sit-walkers-and-the-proxemic-reward)). It was tested on the
validation trips only. Drive it with `python webots/run.py fp_hospital --model models/sit_prox.zip --walkers learned`.

## Problems found and fixed


| Problem | Effect | Fix |
| --- | --- | --- |
| Chair not fully reset between trips (caster angles carried over) | the same trip could end differently | chair restored to its exact starting state before each trip |
| `train.py` skipped training when starting from the 2D policy | it saved the 2D policy unchanged | counts its own steps |
| One warehouse map uses 3 cm pixels | some trips started touching a wall | resampled to 5 cm |
| A hidden or waiting-area person could start on the chair's start | instant crash at 0.1 s | everything added for an event is kept clear of the start and goal |
| Walls in one block, furniture as separate objects | big buildings very slow | walls in 8 m tiles with the furniture inside: about 6x faster, furnished worlds about 60x |
| Smoothness (`mean_jerk`) always 0 | metric missing | measured |

## Limitations

- Fine-tuning in 3D gave small gains: one to three trips out of 140 in the first run, which is within
  luck, and under 10% fewer hard brakes in the run with the SiT walkers.
- The 2D simulator has no spread-out crowd setting, so the 2D and 3D results are not a like-for-like
  comparison.

## References

- **PPO.** Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- **Stable-Baselines3.** Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268), 2021. [Code](https://github.com/DLR-RM/stable-baselines3).
- **Webots.** Michel, O. *Webots: Professional Mobile Robot Simulation.* International Journal of Advanced Robotic Systems 1(1), 39-42, 2004. [cyberbotics.com](https://cyberbotics.com).
