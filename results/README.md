# Results quoted in the docs

The result files behind the numbers in [docs/evals.md](../docs/evals.md), so they can be checked and
compared with a new run. A `.parquet` file has one row per trip with every measure (columns in
[docs/evaluation.md](../docs/evaluation.md)). A `.log` file is what the script printed.

## The latest model

The latest model, `models/sit_prox.zip`, was fine-tuned with walkers trained on real people (SiT) and the
proxemic reward. On the 540 validation trips with those walkers it reached the goal in 539, with no
collisions and 1 timeout, against 538 and no collisions for the first model. The results come from the
retraining run on the AWS machine on 20 November, and `2026-11-20-retrain/` has everything that run
saved, unchanged. The logs of the SiT walker runs and the
two Webots trainings are 11 to 15 MB each, mostly one Trajectron++ warning repeated. The four models it
trained are in `models/`.

| Result file | Policy | Walkers |
| --- | --- | --- |
| `20261120-204843-webots-auto.parquet` | the first model, `models/assist_ppo.zip` | SiT |
| `20261120-210815-webots-rule.parquet` | the cautious hand-written rule | SiT |
| `20261120-212925-webots-r_old_2d.parquet` | `models/r_old_2d.zip`: trained again in 2D, original reward | SiT |
| `20261120-214940-webots-r_prox_2d.parquet` | `models/r_prox_2d.zip`: trained again in 2D, proxemic reward | SiT |
| `20261120-220955-webots-sit_old.parquet` | `models/sit_old.zip`: `r_old_2d` fine-tuned in Webots with the SiT walkers | SiT |
| `20261120-222941-webots-sit_prox.parquet` | `models/sit_prox.zip`: `r_prox_2d` fine-tuned the same way, the latest model | SiT |
| `20261120-224921-webots-auto.parquet` | the first model | rules |
| `20261120-230031-webots-r_old_2d.parquet` | `models/r_old_2d.zip` | rules |
| `20261120-231220-webots-r_prox_2d.parquet` | `models/r_prox_2d.zip` | rules |
| `20261120-232345-webots-sit_old.parquet` | `models/sit_old.zip` | rules |
| `20261120-233512-webots-sit_prox.parquet` | `models/sit_prox.zip` | rules |

Each test's printout is in `eval_<policy>_<learned or rules>.log`, each training's in `train_<model>.log`.
`compare_table.txt` has every run side by side, `tests.log` the test suite run first, `quick_check.log` a
2,000-step training run that checked Webots started on the machine,
and `timeline.txt` and `machine.txt` when each step ran and on what. The commands are in
[docs/evals.md](../docs/evals.md#retraining-with-the-sit-walkers-and-the-proxemic-reward). To compare
the latest model with the first model trip by trip:

    python eval/compare.py results/2026-11-20-retrain/20261120-204843-webots-auto.parquet results/2026-11-20-retrain/20261120-222941-webots-sit_prox.parquet --by crowd

To test the latest model again:

    python webots/evaluate.py --model models/sit_prox.zip --walkers learned --parallel 2

A rerun can differ from these numbers by a few trips, because a trip's result can depend on the trips
run before it in the same Webots ([docs/webots_results.md](../docs/webots_results.md#results)).

## Kept here

| Folder | What | Command | Result |
| --- | --- | --- | --- |
| `2026-11-10-test-set/1-no-rules` | the 3D test set before the rules that keep the chair going | `python webots/evaluate.py` | 533 of 540 reached the goal (6 timeouts, 1 crash). The depot's parquet file was lost, its log is here |
| `2026-11-10-test-set/4-walkers-only` | the same trips with the 7 rules | `python webots/evaluate.py` | 537 of 540 (3 timeouts), no crashes |
| `2026-11-18-final` | the held-back test on seeds 20000 and up: the first model and two drivers without learning | `python webots/evaluate.py --split test --driver auto`, then `--driver route_follower` and `--driver rule` | learned 536 of 540 (1 collision), cautious rule 537 (0), route follower 229 (311) |
| `2026-11-18-test-set` | the same trips with 8 rules (spin fix), the wall brake and the seated rider | `python webots/evaluate.py --parallel 2` | 537 of 540 (3 timeouts, all events trips), no crashes |
| `2026-11-18-override-check` | the rider drives at walls, with and without the brake | `python webots/override_check.py`, then `--no-brake` | brake on: 8 reached a wall, stopped 0.06 to 0.11 m short, none touched it. Brake off: 5 of 8 touched it |
| `2026-11-18-spin-check` | turning on the spot, and what caused it | `python webots/spin_check.py` | 20 of 20 reached the goal, turning on the spot 3% of the time, 92% of it held by the safety stop |
| `2026-11-18-2d` | the 2D assistant on the 2D validation scenes | `python eval/run_episodes.py --scenario all --episodes 20 --driver assist` | 157 of 160 reached the goal, 3 hit a person (corridor seed 10014, hospital seeds 10012 and 10015) |
| `2026-11-20-retrain` | retraining with the SiT walkers and the proxemic reward: 4 new policies, the first model and the cautious rule, on the 540 validation trips, with every log | the commands in [docs/evals.md](../docs/evals.md#retraining-with-the-sit-walkers-and-the-proxemic-reward) | the latest model (`sit_prox`) with SiT walkers: 539 of 540, 0 collisions, 1 timeout. The first model: 538, 0 collisions |

Every 3D test set run uses the same 540 trips (9 test worlds, seeds 10000 to 10019, 3 crowds), so two
runs compare trip by trip. `evaluate.py` saves one file per world, and `all_trips.parquet` in each
folder is those files put together:

    python eval/compare.py results/2026-11-10-test-set/4-walkers-only/all_trips.parquet results/2026-11-18-test-set/all_trips.parquet --by crowd

The held-back test uses the seeds 20000 to 20019 instead, and compares the same way:

    python eval/compare.py results/2026-11-18-final/rule.parquet results/2026-11-18-final/learned.parquet --by crowd
