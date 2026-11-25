# Evaluation: is the chair comfortable, not only safe?

Success and collision rates are close to 100% and 0%, so they no longer tell policies apart. What we
want to measure is whether the chair drives smoothly around people, predicts where they are walking,
and avoids stopping and starting every time someone crosses in front of it. `eval/comfort.py` measures
that from every trip, in 2D and in Webots. `eval/compare.py` compares two runs on the same trips.

## Run it

Comfort columns are added automatically to every results row; nothing extra to switch on.

```bash
# 2D (fast, no Webots): the sweeps and checkpoint selection happen here
python eval/run_episodes.py --scenario floorplans --episodes 20 --driver assist --save-traces
python eval/run_episodes.py --scenario rooms --episodes 20 --driver rule

# Webots (3D): the final checks
python webots/evaluate.py --parallel 4 --model models/assist_ppo_webots.zip --save-traces

# Compare two runs trip by trip, or list many runs side by side
python eval/compare.py data/episodes/A.parquet data/episodes/B.parquet --by scenario
python eval/compare.py --table data/episodes/*.parquet

# Check the metric code itself (no simulator needed)
python tests/test_comfort.py
```

`--save-traces` also writes the chair's speed and turn rate at every step to
`data/traces/<run_id>.parquet`, to plot a speed profile or look at one bad trip.

## The columns

Each trip gets these, next to `success`, `collision`, `mean_jerk` and the rest. "Near" means a person
within 5 m (edge to edge). The first and last 2 s of a trip are not judged (speeding up from standstill,
slowing down to arrive). All thresholds are constants at the top of `eval/comfort.py`.

| Column | Meaning | Good is |
| --- | --- | --- |
| `stops`, `stops_per_min` | Times the chair, after moving (above 0.3 m/s), slowed below 0.1 m/s. The standstill at the start and the stop at the goal are not stops. | low |
| `stops_near_people`, `stops_clear_of_people` | The same, split by whether anyone was within 5 m when it stopped. | low; the second is 0 |
| `stop_go_cycles` | Stops after which it drove on again. | low |
| `flicker_stops` | Stops shorter than 1 s: stopping and going straight away. | 0 |
| `unneeded_stops` | Stops where, had the chair carried on at its speed and heading for 3 s, nobody would have come within 0.5 m. *Hindsight, see below.* | 0 |
| `waiting_near_people_s` | Seconds standing still with someone near: the cost of waiting for a gap. | low |
| `mean_speed_near_people`, `min_speed_near_people` | Speed while people are near, as a share of top speed. 1 means it did not slow down; 0 means it stopped. | high, but not at the price of safety |
| `peak_decel_mps2`, `hard_brakes` | Strongest braking, and the number of separate brakings harder than 1.0 m/s2 (a starting value, tune it). | low |
| `peak_jerk_mps3` | Largest change of acceleration. | low |
| `intrusion_s` | Seconds spent moving closer than 0.5 m to a person. | low |
| `intimate_s`, `personal_s`, `social_s`, `public_s`, `nobody_s` | Seconds with the nearest person in each proxemic zone ([proxemics.md](proxemics.md): ellipses around the chair's 1.1 x 0.65 m outline, 0.45 / 1.2 / 3.6 m, the front stretched up to 2x with speed), or nobody within 12 m. They add up to the trip time. | intimate low |
| `intimate_entries` | Separate times someone came into the intimate zone. | low |
| `intimate_moving_in_s` | Seconds in the intimate zone while the chair was moving, with the person beside or ahead: the part the chair drove into. | low |
| `speed_in_intimate` | Speed while someone is in the intimate zone, as a share of top speed. | low |
| `min_zone_gap_m` | Closest approach in zone metres (0 = touching the outline). | high |
| `proxemic_cost` | What the proxemic reward (`train_assist.py --reward proxemic`) would have taken off over the trip. | low |
| `closest_rel_speed_mps` | How fast chair and person were closing at the moment of closest approach. | low |
| `weaves_per_min` | Left, right, left (or the reverse) within 1.5 s. | low |
| `safety_stops` | Times the 0.4 m safety stop overrode the policy. It is meant as the last resort, so this should be near 0. | 0 |
| `policy_active_share` | Share of the trip when the policy, not plain route following, was in charge. | context |
| `delay_ratio` | Trip time divided by the shortest route at top speed. 1.0 is a trip with nobody in the way. | near 1 |
| `pred_err_1s_m`, `_2s_m`, `_3s_m` | How far the tracker's straight-line prediction was from where a walking person really was 1, 2 and 3 s later. | low |
| `alarms`, `alarm_false_rate` | Predicted collisions raised while the chair was moving, and the share where carrying on would in fact have stayed 0.5 m clear. | low |
| `missed_conflict_s` | Seconds where nobody was flagged but carrying on would have come within 0.3 m. | 0 |

Read them as a profile, not one score. A chair that never moves has no stops and no intrusions; a chair
that ignores people is fast and smooth until it hits someone. Look at `delay_ratio`, the stop columns
and the collision columns together, and prefer a run that is better on some without being worse on others.

### Hindsight numbers are estimates

`unneeded_stops`, `alarm_false_rate` and `missed_conflict_s` ask what would have happened if the chair
had kept going, using where people really went. But people react to the chair: a stopped chair gets
walked past, a moving one gets avoided. So these are estimates of the counterfactual, good for comparing
two policies on the same trips, not a verdict on one trip. The comfort columns (stops, speed, braking,
weaving) are read straight from the chair's own motion and carry no such assumption.

The prediction columns compare the tracker with the simulator's pedestrians, who follow rules we wrote.
They tell us whether prediction improved on this simulator, not on real people.

## Comparing runs

`eval/compare.py A B` matches trips by (world, crowd, seed) and reports, per metric, the mean on each,
the mean difference B minus A with a 95% bootstrap interval, and the number of trips. Only believe a
difference whose interval excludes 0; it labels those "B better" or "B worse". With success near 100%
and 20 trips per world, a gap of one to three trips is noise: use `--episodes 100` for anything you
want to claim.

Use the seeds as before: pick and tune on 10000 to 10199; the final evaluation on 20000 and up is run
once, at the end, on the frozen model.

## A first look (2D, 15 trips: hospital, office, warehouse, seeds 10000 to 10004)

A smoke test of the new code, not a baseline: 5 trips per floor plan is far too few to conclude from.
Run the real baseline before changing anything (see docs/aws_training.md).

| `models/assist_ppo.zip` | Hospital | Office | Warehouse |
| --- | --- | --- | --- |
| Stops per minute | 7.4 | 9.2 | 1.4 |
| Safety stops per trip | 3.8 | 4.0 | 0.2 |
| Hard brakes per trip | 13 | 21 | 5 |
| Peak braking (m/s2) | 8.1 | 8.4 | 3.4 |
| Slowest speed near people (share of top) | 0.00 | 0.00 | 0.37 |
| `delay_ratio` | 1.6 | 1.8 | 1.3 |
| `unneeded_stops` | 0 | 0 | 0 |
| Tracker error 2 s ahead (m) | 0.68 | 0.62 | 0.60 |

`route_follower` (ignores people) has no stops at all, and collides in most of those trips.

Going through two hospital trips stop by stop: nearly every stop began with someone 0.16 to 0.40 m from
the chair, usually with the safety stop firing, and some are flickers (stop at 4.1 s, go, stop again
at 4.6 s). So on this evidence the stopping is not caused by false alarms (`unneeded_stops` is 0 and
`alarm_false_rate` is 10 to 20%). The chair reacts late, ends up almost touching someone, and the
safety stop brakes it dead. That points at reacting earlier and braking gradually, not at predicting
less cautiously. Treat it as a hypothesis until the 100-trip baseline confirms it.

Braking numbers are dominated by the controller, which brakes at once (top speed to 0 in one 0.1 s
step is 12 m/s2). They will only fall if the controller gets a deceleration limit.

## Not built yet

- The rest of the scripted scenario suite. Crossing from the left, from the right and head-on exist
  (`webots/crossing.py`, the same people for every driver). Overtaking, a doorway, and a pedestrian who
  turns or stops mid-crossing do not, and none of it has pass criteria written before looking at results.
- Pedestrian cost: how often people had to change course or wait because of the chair.
- A simulated user who takes over (`sim_user/` is empty), and a small human preference test using the
  first-person view (**F**). Scripted take-overs exist (`webots/override_check.py` and
  `webots/profile_run.py --at-people`, see [evals.md](evals.md#while-the-rider-drives)), but they are
  fixed manoeuvres, not a model of a rider.
- A deceleration limit in the controller, and comfort terms in the reward. Comfort is measured
  (`eval/comfort.py`) but nothing in the controller or the reward uses it yet.
