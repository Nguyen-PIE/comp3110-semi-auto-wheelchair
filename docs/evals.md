# Evals: every test and its result

What each test checks, the command that runs it again, what it ran on, what it gave, and what it is
compared with. The results quoted here are kept in [results/](../results/README.md).

## How the tests are set up

- **Held back.** The 3D trips use worlds and seeds that training never sees ([data.md](data.md#splits)).
  The validation seeds (10000 and up) were also used while the rules that keep the chair going were
  being built, so they are not a clean test. The held-back test ran once, on seeds 20000 and up,
  which nothing was trained, tuned or chosen on, with the first model (`models/assist_ppo.zip`).
- **Baselines.** Each result is compared with the simplest thing that could do the same job: drivers
  without learning for the chair, exact phrases for the rider's own words, coloured tags for the sign
  detector, the brake switched off for the safety brake, the first walk model for the learned walkers.
- **Real and synthetic cases.** Real data is used where it exists: the walk model is trained and tested
  on real people recorded walking round a robot (SiT), and recorded SiT people can be replayed among the
  walkers in a trip. There is no recorded data of a powered wheelchair crossing a busy hospital, so the
  trips, the signs, the riders' requests and the voices are synthetic, made so they can be repeated
  exactly and run in the hundreds.

## Metrics

| Metric | Definition |
| --- | --- |
| Reached the goal | The chair's centre came within 0.5 m of the goal before the time limit (60 s in the corridor and the house, 90 s in most buildings, 120 s in the biggest warehouse), without a collision |
| Collision | The chair (a 0.4 m radius circle) overlapped a person (0.72 m wide with their personal space), or its centre came within 0.4 m of a wall on the floor plan. Split into hit while moving and hit while stopped (still for the last 0.5 s: someone walked into it) |
| Timeout | Neither of the above by the time limit |
| Near miss | Passing within 0.3 m of a person, edge to edge |
| Time to the goal, distance driven | Over the trips that reached the goal, and against the shortest route |
| Comfort | Stops a minute, hard brakes, waiting time, speed near people and more ([evaluation.md](evaluation.md)) |
| Gap to a wall | Lidar distance from the chair's front edge to the nearest wall straight ahead (`guard.py`) |
| Voice | Right command, wrong command (the chair does something not asked), or nothing done |
| Own words | Right answer, a wrong move, or asking back. A wrong move counts as the worst outcome |
| Signs | Sign found, how far its position is from the true one, and the words read exactly right |
| Walk prediction | Mean distance between where people were predicted to be and where they were over the next 4.8 s (ADE), and at the end (FDE) |

## Summary

| Test | Command | Ran on | Result | Baseline | Details |
| --- | --- | --- | --- | --- | --- |
| The latest model: retrained with the SiT walkers and the proxemic reward (3D) | the commands [below](#retraining-with-the-sit-walkers-and-the-proxemic-reward) | 11 runs of the 540 validation trips, SiT walkers and rule walkers | latest model (`models/sit_prox.zip`), SiT walkers: 539 reached, 0 collisions, 1 timeout, 10.3 hard brakes a trip. The proxemic reward alone: no clear difference | the first model: 538, 0 collisions, 11.4 hard brakes a trip. Cautious rule: 536, 0 collisions, 15.5 | [below](#retraining-with-the-sit-walkers-and-the-proxemic-reward) |
| Held-back test of the first model (3D) | `python webots/evaluate.py --split test --parallel 2` | 540 held-back trips: 9 test worlds, seeds 20000 to 20019, 3 crowds | 536 reached the goal (99.3%), 1 collision, 3 timeouts | cautious rule: 537, 0 collisions, 14.5 hard brakes a trip against 10.1. Route follower: 229 (42%), 311 collisions | [below](#held-back-test-the-first-model) |
| Reaching the goal, validation (3D) | `python webots/evaluate.py --parallel 2` | 540 trips: the same worlds, seeds 10000 to 10019 | 537 reached the goal, 0 crashes, 3 timeouts | 533 and 1 crash before the rules that keep the chair going | [below](#the-3d-test-set-validation-seeds) |
| Stopping short of walls while the rider drives at them | `python webots/override_check.py` | 10 hospital trips | the 8 that reached a wall stopped 0.06 to 0.11 m short, none touched it | brake off (`--no-brake`): 5 of 8 touched it | [below](#while-the-rider-drives) |
| Not driving into people while the rider drives at them | `python webots/profile_run.py fp_hospital --open-world --steps 1200 --at-people` | 4 take-overs of 20 s | 0 people hit | brake off: 3 hit | [below](#while-the-rider-drives) |
| Turning on the spot | `python webots/spin_check.py` | 20 trips in the house never trained on, learned walkers | on the spot 3% of the time, 92% of that held by the safety stop | no run from before the fix is kept | [below](#turning-on-the-spot) |
| Finding a named room without the floor plan | `python webots/open_eval.py` | 10 hospital trips | 10 of 10 reached, median 6 times as long | the same trips with the floor plan | [webots_open_world.md](webots_open_world.md#results) |
| Reading signs | `python webots/sign_eval.py` | 1233 views of 160 hospital signs | tag found 71%, placed within 0.09 m (median), words read exactly 36% (58% at 3 to 5 m) | none | [webots_open_world.md](webots_open_world.md#results) |
| A sign detector instead of tags | `python webots/sign_detector.py evaluate` | 480 signs in trained buildings, 176 in one never trained on | finds 91% and 90% | coloured tags: 49% and 45% | [webots_open_world.md](webots_open_world.md#results) |
| Own words to a command | `python webots/interpret_eval.py all` | 114 test requests | 92% right, 0 wrong moves | exact phrases alone: 25% | [voice.md](voice.md#choosing-the-model) |
| Exact voice commands | `python webots/voice_eval.py run --tts` | 1,680 commands: 6 computer voices, 4 noise levels | 88% understood in quiet, 63% at 5 dB, 0 wrong commands | none | [voice.md](voice.md#exact-commands-without-the-model) |
| Predicting where people walk (learned walkers) | `python webots/walk_model/offline_eval.py --scene sit` | SiT's 14 test scenes, 572 real paths within 3 m of the robot | 0.54 m mean error as used in the sim | the first model (trained on ETH/UCY): 17 to 25% further off | [walk_model/README.md](../webots/walk_model/README.md#training-on-sit-people-walking-round-a-robot) |
| Comfort: stops, braking, speed near people | `python eval/compare.py A.parquet B.parquet` | any two runs of the same trips | measured for every trip | the earlier run | [evaluation.md](evaluation.md) |
| The 2D assistant | `python eval/run_episodes.py --scenario all --episodes 20 --driver assist` | 160 trips: 8 scenarios, seeds 10000 to 10019 | 157 of 160 reached the goal, 3 hit a person (160 of 160 on 18 October) | route follower 96 of 160, cautious rule 160 of 160 (18 October) | [2d-simulator/assistant.md](../2d-simulator/assistant.md#results) |
| The code | `python -m pytest` | 179 tests, no simulator needed | all 179 pass | | `tests/` |

## Held-back test: the first model

Run once on 18 November, on seeds 20000 to 20019 in the same 9 test worlds and 3 crowds (540 trips),
with the first model, `models/assist_ppo.zip`, before the retraining below. Nothing was trained, tuned or chosen on these seeds. The same
trips were driven by the learned assistant and by two drivers without learning: the route follower,
which follows the shortest route and ignores people, and the cautious hand-written rule that the policy
first learned from. The learned assistant and the rule both run inside the same safety rules.

| 540 held-back trips | Route follower (ignores people) | Cautious rule | Learned assistant |
| --- | --- | --- | --- |
| Reached the goal | 229 (42%) | 537 (99.4%) | 536 (99.3%) |
| Collisions | 311 | 0 | 1 |
| Timeouts | 0 | 3 | 3 |
| Time to the goal (trips that got there) | 12.1 s | 22.1 s | 20.5 s |
| Hard brakes a trip | 0.6 | 14.5 | 10.1 |
| Stops a minute | 0.0 | 8.2 | 5.3 |
| Seconds inside someone's personal space a trip | 1.6 | 4.1 | 4.7 |
| Near misses a trip | 1.11 | 1.65 | 1.68 |

The learned assistant reached the goal in all 180 spread and all 180 busy trips, and 176 of 180 events
trips. Paired trip by trip against the cautious rule (`eval/compare.py`, 95% bootstrap intervals), it reached
the goal 1.4 s sooner (0.8 to 2.0 s), with 4.5 fewer hard brakes a trip (3.8 to 5.1) and 2.9 fewer stops
a minute (2.5 to 3.4), and the same success within 1%. It also used the safety stop more (2.9 against
2.0 times a trip) and spent longer close to people. Its one collision was in the hospital's events crowd
(seed 20000): a person closing at 1.7 m/s while the chair was braking.

So learning made the chair smoother and quicker than the rule it started from, not safer. The safety
comes from the rules around the policy, which stay plain code on purpose. The files are in
[results/2026-11-18-final](../results/README.md).

## The 3D test set: validation seeds

These are the same worlds as the held-back test, on the validation seeds, which were used while the
rules were built. `evaluate.py` runs 20 trips in each of the 9 test worlds without a window: the 7 training buildings
with furniture never trained on, and a house never trained on, as it is and furnished. Each trip runs
three times, with people spread over the building (the main score), with people crowded along the
route (a stress test), and with each building's events (an emergency trolley, a forklift, a meeting
ending). The chair is reset to exactly the same state before each trip, so every version gets exactly
the same trips.

| 18 November, 8 rules, wall brake, seated rider | spread | busy | events | all |
| --- | --- | --- | --- | --- |
| Reached the goal | 180 of 180 | 180 of 180 | 177 of 180 | 537 of 540 (99.4%) |
| Crashes | 0 | 0 | 0 | 0 |
| Timeouts | 0 | 0 | 3 | 3 |

| The same 540 trips over time | reached the goal | timeouts | crashes |
| --- | --- | --- | --- |
| 10 November, before the rules that keep the chair going | 533 | 6 | 1 |
| 10 November, 7 rules | 537 | 3 | 0 |
| 18 November, 8 rules (spin fix), wall brake, seated rider | 537 | 3 | 0 |

The 3 timeouts are events trips in which the chair waited for a way that did not open: corridor seeds
10017 and 10018 (two-way traffic) and hospital seed 10006. What each rule changed and the comfort
numbers are in [webots_results.md](webots_results.md). The files are in
[results/2026-11-18-test-set](../results/README.md).

## Retraining with the SiT walkers and the proxemic reward

Run on 20 November on the AWS machine (16 vCPUs, 6.5 hours), on the validation seeds 10000 to 10019
in the 9 test worlds and 3 crowds (540 trips a run). The policy was trained again in 2D with the same
settings twice, once with the old reward and once with the proxemic reward ([proxemics.md](proxemics.md)),
and each was then fine-tuned in Webots for 110,000 steps with the walkers trained on SiT. The four new
policies, the first model and the cautious rule were tested with the SiT walkers (`--walkers learned`),
and the four and the first model again with the rule walkers. Each policy was trained once (one seed).

| 540 trips, SiT walkers | First model | Cautious rule | 2D, old reward | 2D, proxemic | Fine-tuned, old | Fine-tuned, proxemic |
| --- | --- | --- | --- | --- | --- | --- |
| Reached the goal | 538 | 536 | 537 | 539 | 539 | 539 |
| People hit | 0 | 0 | 2 | 1 | 1 | 0 |
| Timeouts | 2 | 4 | 1 | 0 | 0 | 1 |
| Time to the goal | 21.7 s | 23.1 s | 22.6 s | 22.5 s | 22.3 s | 22.0 s |
| Hard brakes a trip | 11.4 | 15.5 | 11.0 | 11.0 | 10.6 | 10.3 |
| Weaves a minute | 0.12 | 0.31 | 0.18 | 0.32 | 0.16 | 0.31 |
| Share of the trip with someone in the intimate zone | 41% | 40% | 38% | 39% | 38% | 37% |
| Seconds a trip driving into someone's intimate zone | 5.8 | 5.0 | 5.4 | 5.5 | 5.4 | 5.0 |

With the rule walkers, the first model reached 536 (1 person hit, 3 timeouts, 10.6 hard brakes a trip)
and the fine-tuned proxemic policy 538 (no person hit, 1 wall hit, 1 timeout, 10.2 hard brakes). The
first model reached 537 with no collision on the same trips on 18 November on another computer.
Reruns of the same 540 trips have differed by a few trips before (535 to 538 over five reruns,
[webots_results.md](webots_results.md#results)), because a trip's result can depend on the trips run
before it in the same Webots.

Compared trip by trip (95% bootstrap intervals):

- **The proxemic reward made no clear difference.** Against the same training with the old reward, most
  measures moved by less than 5%, with intervals that include no change. The clear changes: after the
  fine-tune with the SiT walkers, 6.9% less time driving into someone's intimate zone (1.7 to 11.9%),
  in 2D with the rule walkers, 6.2% fewer stops a minute (0.3 to 11.9%), and in every pair the chair
  weaved about twice as often (0.16 against 0.31 a minute after the fine-tune, +0.06 to +0.26). The pass
  bar in [proxemics.md](proxemics.md) (25% less time with someone in the intimate zone) is not met.
- **The fine-tune with the SiT walkers helped a little.** With the SiT walkers, the fine-tuned proxemic
  policy braked hard 9.6% less than the first model (4.8 to 14.7%) and drove into someone's intimate zone
  12.8% less (6.7 to 19.4%), with no clear change in time to the goal. The fine-tuned old-reward policy
  braked hard 7.0% less (1.9 to 11.8%), so most of the gain comes from the fine-tune. With the rule
  walkers the gain is smaller and not clear (3.8% fewer hard brakes, -8.7 to +0.5%).
- **People are close whatever drives the chair.** Someone is in the chair's intimate zone for about 40%
  of every trip, for the cautious rule too: about 45% in the busy and events crowds and 25% in the spread
  crowd. The chair is moving with them beside or ahead for a little over half of that time, so a change
  to the reward can only move part of it.
- **Collisions** (0 to 2 in 540 for every policy) are too few to tell the policies apart.

The fine-tuned proxemic policy, `models/sit_prox.zip`, is the latest model: with the SiT walkers it had
no collisions, braked hard 9.6% less and drove into someone's intimate zone 12.8% less than the first
model. It was tested on the validation trips only, so the held-back result above is the first model's,
and `run.py` starts with the first model unless given `--model models/sit_prox.zip`. The result files
are in [results/2026-11-20-retrain](../results/README.md#the-latest-model). The commands:

```bash
python training/train_assist.py --reward old      --seed 0 --out models/r_old_2d.zip
python training/train_assist.py --reward proxemic --seed 0 --out models/r_prox_2d.zip
python webots/train.py --start-from models/r_old_2d.zip  --reward old      --walkers learned --envs 4 --timesteps 100000 --out models/sit_old.zip
python webots/train.py --start-from models/r_prox_2d.zip --reward proxemic --walkers learned --envs 4 --timesteps 100000 --out models/sit_prox.zip
python webots/evaluate.py --model models/sit_prox.zip --walkers learned --parallel 2   # and so on for each policy and walkers
```

The result files and logs are not kept in Git.

## While the rider drives

The rider takes over (**O**) and holds the up arrow, as a rider who never lets go would.
`override_check.py` turns the chair until a wall is 1 to 3 m straight ahead, then drives at it for
6 s. `profile_run.py --at-people` drives straight at the nearest person.

| 10 hospital trips: the rider holds forward at a wall | brake on | brake off (`--no-brake`) |
| --- | --- | --- |
| Reached a wall (the others turned into open space) | 8 | 8 |
| Touched it (centre 0.40 m from the wall) | 0 | 5 |
| Closest the centre came to a wall | 0.45 m | 0.40 m |
| Gap between the chair's front and the wall (lidar) | 0.06 to 0.11 m | -0.09 to 0.12 m, median 0.04 m |
| Counted as a wall hit by the trip's rule | 0 | 0 |

A gap below 0 is the chair pressed against the wall at an angle. The last row is the gap in the trip's
rule described under [known gaps](#known-gaps).

| 4 take-overs of 20 s: the rider drives at the nearest person (16 November) | brake on | brake off |
| --- | --- | --- |
| People hit | 0 | 3 |

The files are in [results/2026-11-18-override-check](../results/README.md).

## Turning on the spot

`spin_check.py` counts the steps where the chair moves forward slower than 0.1 m/s while turning
faster than 0.3 rad/s, and which part of the code was in charge at the time.

| 20 trips, `fp_house_test`, learned walkers, the fine-tuned policy | |
| --- | --- |
| Reached the goal | 20 of 20 |
| Turning on the spot | 7 s of 216 s driving (3%) |
| While the safety stop held the chair (someone right ahead) | 6.7 s (92%) |
| The policy asking for no speed | 0.2 s (3%) |
| Getting out of a crush, or the route pointing behind | 0 s |
| Steering flipping from side to side while stopped | 16 times in 20 trips |

Rule 8 in [webots_results.md](webots_results.md) was added for the turning that came from getting out of
a crush and from new routes back the way the chair came. None is left: what remains is the chair
turning in place while someone stands right in front of it. No run from before the change is kept
here. The files are in [results/2026-11-18-spin-check](../results/README.md).

## Reproducing

Every command above runs from the `G19` folder after the [install](../README.md#install), and the 3D
tests need the worlds made first (`python webots/make_world.py all --variants 3`, which gives the same
worlds as the ones in Git). A trip is set by its world, seed and crowd, and the chair is reset to
exactly the same state before each one, so a rerun drives the same trips, and `eval/compare.py` pairs
two runs trip by trip. A rerun's totals can still differ by a few trips, because a trip's result can
depend on the trips run before it in the same Webots ([webots_results.md](webots_results.md#results)). On a laptop with 2 cores, the 540 trips take about 45 minutes with
`--parallel 2`, `override_check.py` about 2 minutes and `spin_check.py` about 1 minute.
`interpret_eval.py` needs the course gateway key in `.env`, and the SiT walk tests need SiT
([data.md](data.md#how-the-data-is-prepared)).

## Known gaps

- A trip counts a wall hit when the chair's centre is closer than 0.40 m to a wall on the floor plan
  (5 cm grid). A solid Webots wall stops the chair at 0.40 m, so a chair that touches a wall is not
  counted. `override_check.py` counts touching separately, and the test set's 0 wall hits should be
  read with this in mind.
- The 2D assistant hit 3 people in 160 trips on 18 November, against none on 18 October. The rules
  that keep the chair going and the walkers were changed and tuned in 3D in between, and the 2D test
  was not run in that time. The cause has not been found.

- The latest model (`models/sit_prox.zip`) was tested on the 540 validation trips, not on the held-back
  seeds, which were used once, for the first model.

## Not tested

- Real people and a real chair. Everything here is in simulation, with simulated people.
- Real voices: the exact-command numbers are from computer voices, which are clearer than people.
  `voice_eval.py record` measures real voices the same way.
- Riders in their own words: the 231 requests were written by the team, not collected from wheelchair
  users.

## References

- **Bootstrap intervals** (`eval/compare.py`, 5,000 resamples). Efron, B. and Tibshirani, R. J. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
- **Webots.** Michel, O. *Webots: Professional Mobile Robot Simulation.* International Journal of Advanced Robotic Systems 1(1), 39-42, 2004. [cyberbotics.com](https://cyberbotics.com).
