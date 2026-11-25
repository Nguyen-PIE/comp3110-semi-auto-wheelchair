# Learned walkers: Trajectron++ trained on real people

`--walkers learned` makes the ordinary people in the 3D simulator step the way a published
trajectory model says real people would, instead of following the hand-written "social force"
rules. Pets, children, trolleys and forklifts keep their rules. The default is still
`--walkers rules`, so training and testing stay comparable with earlier results.

```
python webots/run.py fp_hospital --walkers learned
python webots/evaluate.py --walkers learned          # the test set, with the learned crowd
```

## What it is

- **The model.** [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) (Salzmann et al.,
  ECCV 2020, MIT licence). It's one of the most-cited models for predicting where people walk. It
  looks at the last 3.2 s of a person and everyone within 3 m of them, and suggests several
  possible ways they could carry on.
- **The data.** ETH/UCY: about 1,500 real people filmed from above in 5 public places. They are a
  university entrance (eth), a hotel front (hotel), a university campus (univ) and two shop fronts
  (zara1, zara2). Positions are in metres, every 0.4 s.
- **Training.** We left out `zara1`, so we can test on people the model never saw. Training takes
  about 4 minutes per pass on a 4-core CPU, and no GPU is needed. The model included here
  (`models/walk_trajectron/`) had 30 passes. The paper used 100 (see "Next steps").

## How it moves people (webots/learned_walk.py)

1. Every 0.4 s, the model gets the last 3.2 s of everyone in the scene. The chair is included as if
   it were a person, so people react to it. The model suggests 20 paths for each walker, 2.4 s
   ahead.
2. The recordings have no floor plans or destinations, so the model doesn't know where anyone is
   going. For each person we keep the suggestion that best heads toward the next corner of their
   route. Suggestions that go through a wall or bump into someone are ruled out, and suggestions
   close to what the person was already doing are preferred, so nobody dithers.
3. Each 0.1 s step, the person follows that suggestion. The rules in `envs/pedestrians.py` still
   run on top, as with the old walkers: noticing the chair (attentive or distracted), stepping
   aside, keeping off walls, and speeding up and turning smoothly.

The model's random choices come from the trip's seed, so a seed still replays the same way. This
is checked by `tests/test_learned_walk.py`.

## Results (model after 30 passes)

**Offline: 2,356 real people at zara1, a place the model never saw.** We predict 4.8 s of walking,
then measure the average distance from the real path (ADE) and the distance at the end (FDE), in
metres. Lower is better.

| method | ADE | FDE | knows the destination? |
|---|---|---|---|
| keep walking straight on | 0.43 | 0.95 | no |
| old rules (social force) | 0.30 | 0.31 | yes: aimed straight at where the person really ended up |
| learned, as used in the sim | 0.39 | 0.81 | partly: the direction it picks for the next 1.2 s |
| learned, most likely path | 0.49 | 1.09 | no |
| learned, best of 20 (the usual score in papers) | 0.26 | 0.55 | no |

How to read this. The old rules look best only because this test hands them the person's true end
point, which makes their FDE almost free. On real people, the model's best suggestions (0.26) are
closer than either baseline, so the realistic paths are among its suggestions. But its single
most likely path is not yet better than walking straight on, which says it is under-trained. The
paper reports about 0.15 / 0.33 for best of 20 at zara1 after 100 passes.

**Without Webots (`compare_in_sim.py`, hospital, 3 trips of 90 s, 20 people, chair parked):**

| | real people | rules | learned |
|---|---|---|---|
| walking speed, m/s | 1.06 | 0.89 | 0.81 |
| turning, deg/s (median) | 3.7 | 6.5 | 8.2 |
| distance to nearest person, m (median) | 0.78 | 2.24 | 2.17 |
| destinations reached per person per minute | n/a | 0.79 | 0.66 |
| steps inside a wall | n/a | 0 | 0 |
| time per 0.1 s step, ms | n/a | 0.5 | 6.1 |

**In Webots (fp_hospital, seeds 10000 to 10009, the 2D-trained assistant driving):** the chair
reached the goal 10/10 with both crowds. Its average trip was 25.2 s with the rules crowd and
28.4 s with the learned crowd. The learned people walk a little slower and get in the way more.

## Files

- `get_trajectron.sh` downloads Trajectron++ into `third_party/` (`setup.sh` runs it) and applies
  `trajectron_compat.patch`. The patch has the few fixes newer Python, NumPy and PyTorch need.
- `train.sh [passes] [held-out place, or sit / sitmix]` prepares the data, trains, exports and tests.
- `sit_to_trajectron.py <SiT folder> [--with-eth]` converts SiT for training (see above).
- `export.py` saves a training checkpoint as `models/walk_trajectron/walk_model.pt`: just the
  numbers, which load with PyTorch's safety check on.
- `offline_eval.py` runs the zara1 test above, and `compare_in_sim.py` runs the walking stats above.

## Training on SiT: people walking round a robot

ETH/UCY was filmed from above in open plazas, with no robot in it. **SiT** ("Socially Interactive
Trajectory", Hanyang University's SPA Lab, NeurIPS 2023; [site](https://spalaboratory.github.io/SiT/),
[code](https://github.com/SPALaboratory/SiT-Dataset)) was recorded from a Clearpath Husky robot
driven through crowds in 10 kinds of place, including a corridor, a hallway, a lobby and a
cafeteria. Every recording has the robot in it, so a model trained on it learns how real people
move round a robot. That's what our simulated people need. The robot (about 1 m long, walking
pace) is a fair stand-in for the wheelchair.

The model in `models/walk_trajectron/` is already trained on SiT (100 passes, see the results
below), so you only need these steps to train it again.

1. **Get the data.** SiT publish every scene's walking tracks (27 MB, all 65 scenes) on their
   GitHub, so the full 300 GB download isn't needed:
   `bash webots/walk_model/get_sit_from_github.sh ~/sit_data`. (With the full download on a drive,
   `python webots/walk_model/sit_to_trajectron.py <drive folder>` reads its `label_3d/` and
   `ego_trajectory/` folders directly instead.) The licence is CC BY-NC-ND 4.0: fine for a
   university project, but don't put the dataset in the repository (share the trained model, not
   the data).
2. **Convert it**: `python webots/walk_model/sit_to_trajectron.py ~/sit_data/trajectory_txt`. It
   keeps every 4th frame (SiT has 10 a second, the model works in 0.4 s steps), adds the robot as
   track 0 (as `learned_walk.py` adds the chair), splits people's tracks where they were out of
   sight, and uses SiT's own split of scenes (`sit_split.json`: 37 to train, 14 to check, 14 to
   test). Add `--with-eth` to train on SiT plus ETH/UCY ("sitmix"): more people, but fewer of them
   near a robot.
3. **Train**: `bash webots/walk_model/train.sh 100 sit` (or `sitmix`). About an hour on a plain CPU.
   It saves the model to `models/walk_trajectron/` (this replaces the current model, so copy that
   folder first if you want to keep it), then runs `offline_eval.py --scene sit` on SiT's test
   scenes. For SiT the robot itself isn't scored, and the people within 3 m of it get their own
   lines, which are the numbers that matter for us. More than about 100 passes doesn't help: SiT
   is small (65 recordings of 20 s), so a longer run starts memorising them.
4. **Watch it**: `python webots/run.py fp_hospital --walkers learned --tracks`.
5. **Train the chair with them**: `python webots/train.py --envs 12 --timesteps 1000000 --walkers learned`.

**Results of the bundled SiT model** (100 passes, 13 Nov 2026; `offline_eval.py --scene sit`, SiT's
14 test scenes, 5522 real 4.8 s paths, of which 572 were within 3 m of the robot):

| ADE / FDE (m), people within 3 m of the robot | SiT model | old ETH/UCY model |
|---|---|---|
| learned (steered to the destination, as in the sim) | 0.54 / 0.97 | 0.65 / 1.23 |
| learned, most likely path | 0.69 / 1.33 | 0.87 / 1.69 |
| learned, best of 20 | 0.35 / 0.66 | 0.46 / 0.83 |
| (rules, which are told the exact end point) | 0.44 / 0.48 | |

So training on SiT made the predictions 17 to 25% closer to real people. The rules row isn't a fair
fight (it's handed where the person really ended up), but the model's best of 20 still beats it on ADE.

In Webots (fp_hospital_test, 30 trips each, the chair trained among rule-based people): rules 28/30
reached the goal, learned 27/30, no one hit in either. But among the learned people the chair stopped
about twice as often (12.7 vs 7.0 stops a minute) and braked hard more (21 vs 13 a trip): they are less
predictable to a chair that has only practised among the rules, which is why it should be trained
among them.

Tested here only on made-up data in SiT's file layout (the real data needs the form): converting,
preparing, 5 training passes, export, `offline_eval.py --scene sit` and `tests/test_learned_walk.py`
with the exported model all ran.

## Next steps

- Train the full 100 passes on the AWS machine (`bash webots/walk_model/train.sh 100`, a few hours
  on 16 cores), then re-run `offline_eval.py` and `compare_in_sim.py`.
- The recordings are of open plazas and pavements. Corridors and doorways are new to the model,
  which is why the floor plan still decides the route.
- Once the model is trained, fine-tune the chair's assistant against learned crowds (`train.py`
  would need a `--walkers` option too).

## References

- **Trajectron++.** Salzmann, T., Ivanovic, B., Chakravarty, P. and Pavone, M. *Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data.* ECCV 2020. [Code](https://github.com/StanfordASL/Trajectron-plus-plus).
- **ETH.** Pellegrini, S. et al. *You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking.* ICCV 2009.
- **UCY.** Lerner, A., Chrysanthou, Y. and Lischinski, D. *Crowds by Example.* Computer Graphics Forum, 2007.
- **SiT.** Bae, J. et al. *SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots.* NeurIPS 2023 Datasets and Benchmarks. [Site](https://spalaboratory.github.io/SiT/), [code](https://github.com/SPALaboratory/SiT-Dataset). Licence: CC BY-NC-ND 4.0.
- **Social force model.** Helbing, D. and Molnár, P. *Social Force Model for Pedestrian Dynamics.* Physical Review E 51(5), 4282-4286, 1995. The old rules the learned walkers replace.
