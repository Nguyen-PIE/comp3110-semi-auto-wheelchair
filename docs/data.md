# Data: where it comes from, how it is split, and what is kept

Every dataset the project uses, how it is prepared, its licence and how that is handled, which part
is used for training and which for testing, its known quality problems, and what is kept in Git. The
tests themselves are in [evals.md](evals.md).

## Sources

| Data | From | Licence | In Git | Used for |
| --- | --- | --- | --- | --- |
| Hospital, office and warehouse floor plans | [AutoAILab wheelchair-navi](https://github.com/AutoAILab/wheelchair-navi), a fixed commit | none stated | no: `setup.sh` downloads them (`scripts/get_maps.sh`) | the 2D and 3D buildings |
| Depot and warehouse3 floor plans | Nav2's simulation maps | Apache 2.0 | no: `make_world.py` downloads them and checks each file's SHA-256 | 3D training buildings |
| Small warehouse and small house floor plans | AWS RoboMaker's simulation worlds | MIT-0 | no: `make_world.py` downloads them and checks each file's SHA-256 | a 3D training building, and the test building never trained on |
| The 40 Webots worlds and their furnished floor plans | made from those plans by `webots/make_world.py` | ours | yes (`webots/worlds/`), so the worlds open without setup | every 3D trip |
| ETH/UCY pedestrian tracks (5 public places, filmed from above) | research datasets shared with Trajectron++ (`webots/walk_model/get_trajectron.sh`) | not stated | no | the first learned walker model |
| [SiT](https://spalaboratory.github.io/SiT/): people walking round a robot (Hanyang University, NeurIPS 2023) | by Google Form, or its tracks from its GitHub | CC BY-NC-ND 4.0 | no: `data/sit/` is ignored, only the trained model is in Git | the learned walkers, and recorded walkers replayed in the simulator |
| 231 voice requests in riders' own words, each with its screen and right answers | written for this project (`webots/voice_requests.jsonl`) | ours | yes | choosing the model for own words |
| Spoken commands | espeak-ng's 6 English voices, made on each run, and our own voices (`voice_eval.py record`) | | no: `data/voice/` is ignored, as recordings of people's voices | measuring exact commands |
| Sign pictures for the sign detector | drawn by `webots/signs.py` in the worlds, each labelled by where its sign is | ours | no: remade by `sign_detector.py collect` | training and testing the detector |

## How the data is prepared

Every step is a script, so the data can be made again from nothing. Downloads are pinned to a tested
version (a fixed commit, or a SHA-256 checked on every download).

| Data | Steps |
| --- | --- |
| Floor plans | `bash scripts/get_maps.sh` (run by `setup.sh`) fetches the three wheelchair-navi plans at a fixed commit. `python webots/make_world.py all --variants 3` fetches the Nav2 and RoboMaker plans, builds walls from each plan, adds furniture from a fixed seed for each copy (`_v1` to `_v3` for training, `_test` for testing), the signs, the directory and the rider. Run again, it gives the same worlds as the ones in Git. Your own plans go through `scripts/import_floorplan.py`, which turns dark pixels into walls and prints the building size and any door narrower than the chair, to check the scale |
| SiT | `bash webots/walk_model/get_sit_from_github.sh` (its published tracks, 27 MB) or `python scripts/download_sit.py` (the whole dataset). `python webots/walk_model/sit_to_trajectron.py` turns the labels into training data with the scene split in `webots/walk_model/sit_split.json`, and `python scripts/sit_tracks.py` into replay tracks, leaving out seated people |
| ETH/UCY and the walk model | `bash webots/walk_model/get_trajectron.sh` (run by `setup.sh`) fetches Trajectron++ and its data at a fixed commit, with a patch for newer libraries. `bash webots/walk_model/train.sh` trains, and `export.py` saves the model the simulator loads |
| Voice requests | written by hand in `webots/voice_requests.jsonl`, one request a line with the screen and the right answers. `interpret_eval.py` splits them in half by kind with a fixed seed |
| Sign pictures | `python webots/sign_detector.py collect` drives through the worlds, saves pictures and labels each sign from where it is in the world. The depot is kept as a test building, the rest split 80/20 with a fixed seed |
| Results | each `evaluate.py` run writes one row per trip with its run id, world, seed and crowd. The runs the docs quote are copied to `results/` |

## Data quality

| Problem | How it is handled |
| --- | --- |
| Synthetic data: the trips, signs, requests and voices are made, not recorded, so real use will be harder | Said wherever a result is quoted. Real data is used where it exists: the walk model is trained and tested on real people (SiT), and real people can be replayed in a trip (`--replay`) |
| Floor plan pictures: text and door-swing arcs turn into walls, and a wrong scale changes everything | The guide says to erase them first, and the import prints the building size and narrow doors to check |
| Outdoor walking data (ETH, UCY) is faster than people indoors | The walk model was trained again on SiT, recorded round a robot, which made its predictions 17 to 25% closer |
| SiT's labels: a wrong reading would shift every track | The converter was matched to the label format in SiT's own files, and prints the median walking speed as a check |
| Hand-written voice requests may be easier than real riders | The rule for choosing the model was written before any model was tried, and the test half was used once |
| Sign labels come from the world itself, so they are never wrong, but the pictures are rendered, not photographs | Tested on a building the detector never saw. Real photographs are not tested |
| The validation seeds were used while the rules were built | The held-back test uses seeds 20000 and up, which nothing was tuned on ([evals.md](evals.md#held-back-test-the-first-model)) |
| Two runs from 10 November lost some files on a laptop | Noted in `results/README.md`, and the quoted runs are now kept in Git |

## Splits

Nothing a model is tested on is used to train or tune it:

| | Train or choose | Test |
| --- | --- | --- |
| 3D trips | 28 worlds: 7 buildings and their furnished copies `_v1` to `_v3`, seeds 0 to 9999 | 7 copies with furniture never trained on (`_test`) and a building never trained on (`fp_house`, `fp_house_test`), seeds 10000 to 10199 (`webots/splits.py`) |
| 2D trips | seeds 0 to 9999 | seeds 10000 to 10049 |
| Held-back test | | seeds 20000 and up, run once on 18 November with the first model (`evaluate.py --split test`). The latest model was chosen and tested on the validation seeds only |
| Model for own words | 117 requests, half of each kind, and the rule for choosing written before any model was tried | the other 114 |
| Learned walkers | ETH/UCY without `zara1`, then SiT's own split (37 scenes) | `zara1`, then SiT's 14 test scenes |
| Sign detector | 369 pictures of 4 buildings | new pictures of those buildings, and the depot, never trained on |

Test trips never get the random variety training adds (busier crowds, a slower chair, lidar noise),
and the chair is reset to exactly the same state before each one, so every policy gets exactly the
same trips and two runs can be compared trip by trip (`eval/compare.py`).

## Licences and privacy

| Issue | How it is handled |
| --- | --- |
| wheelchair-navi's floor plans have no licence | They are downloaded at setup, not kept in Git. The furnished plans made from them for the worlds are only in the course's private GitLab. A public copy of the repository would make them at setup instead |
| SiT is CC BY-NC-ND 4.0: non-commercial, credit, no shared changes | Used for non-commercial coursework, with SiT credited in the docs. The data stays out of Git. The walk model trained on it is only in the course's private GitLab, and should be swapped for one trained on ETH/UCY (`train.sh`) before any public copy |
| ETH/UCY state no licence | Used for research only, as published with Trajectron++, and not kept in Git |
| Recordings of people's voices | Only team members' own voices, kept on their own laptops (`data/voice/` is ignored) |
| The course gateway key | Kept in `.env`, which is ignored. `setup.sh` makes the file with the key empty |
| What goes to the language model | Text only: what Vosk wrote down on the laptop and what is on the screen. The audio never leaves the laptop |

## What is kept

| Kept in Git | Not in Git, and why |
| --- | --- |
| The code, the 40 worlds, the voice requests | downloaded libraries and floor plans (`third_party/`): remade by `setup.sh` at fixed versions |
| The policies the code loads by default (`models/assist_ppo.zip`, `models/assist_ppo_webots.zip`) and the walk model (`models/walk_trajectron/`) | other models and training runs: large, and remade by the scripts that wrote them |
| The results quoted in the docs (`results/`, a few hundred KB), apart from the retraining runs | every other result (`data/episodes/`, `data/traces/`, `data/steps/`): one file per run, many of them. The retraining runs of 20 November ([evals.md](evals.md#retraining-with-the-sit-walkers-and-the-proxemic-reward)) are among them, and are remade by the commands given there |
| | SiT (`data/sit/`): its licence does not allow sharing it |
| | voice recordings (`data/voice/`): people's own voices |
| | the course gateway key (`.env`): `setup.sh` makes the file with the key empty |

Each run of `evaluate.py` saves one row per trip to `data/episodes/<run_id>.parquet`, with the world,
seed, crowd, result, time, distance and comfort measures ([evaluation.md](evaluation.md)). The runs
the docs quote are copied to `results/` with a README that says what each file is, because earlier runs
were lost from the laptops they ran on.

## References

- **SiT.** Bae, J. et al. *SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots.* NeurIPS 2023 Datasets and Benchmarks. [Site](https://spalaboratory.github.io/SiT/), [code](https://github.com/SPALaboratory/SiT-Dataset). Licence: CC BY-NC-ND 4.0.
- **ETH.** Pellegrini, S. et al. *You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking.* ICCV 2009.
- **UCY.** Lerner, A., Chrysanthou, Y. and Lischinski, D. *Crowds by Example.* Computer Graphics Forum, 2007.
- **Floor plans.** AutoAILab, [wheelchair-navi](https://github.com/AutoAILab/wheelchair-navi) (no licence stated), Nav2 simulation maps (Apache 2.0) and AWS RoboMaker small house and small warehouse worlds (MIT-0).

The models, simulators and libraries are referenced on the pages that use them, listed in the
[README](../README.md#built-with).
