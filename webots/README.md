# Webots (3D) version of the simulator

The same chair, people and buildings as the 2D simulator, in 3D with real wheels, physics and a
laser scanner (lidar). The assistant's code and the policy trained in 2D run here unchanged.

## Install (once per computer)

From the `G19` folder:

```bash
bash setup.sh                                  # the Python environment, libraries and speech model
bash webots/install_webots.sh                  # Webots
conda activate wheelchair                      # in every new terminal
python webots/make_world.py all --variants 3   # build the worlds, about 1 to 2 minutes
```

## Why Webots, and what this adds

We planned to use Isaac Sim for 3D, but it needs an NVIDIA graphics card, and neither our laptops
nor the cloud machine we were given (an AWS c6a.4xlarge: 16 CPU cores, no graphics card) have one.
Webots runs on an ordinary CPU, on macOS, Windows and Linux, and can run without a screen on the
cloud machine, so we train there.

The 3D simulator reuses the 2D code instead of replacing it:

- `make_world.py` turns the 2D floor plans into 3D worlds, so both simulators use the same building
  in the same coordinates.
- `bridge.py` answers the questions the assistant asks the 2D world (where is the chair, where are
  the people, how far are the walls) from the 3D world instead, so `assist/`, `baselines/` and
  `training/` run as they are.
- `train.py` fine-tunes the 2D policy in 3D, starting from `models/assist_ppo.zip`.
- `evaluate.py` is the 3D test set: the same trips for every policy, saved like `eval/run_episodes.py`.

Every test and its result is in [docs/evals.md](../docs/evals.md). In short, the latest model, fine-tuned in
3D with walkers trained on real people (SiT) and the proxemic reward (`models/sit_prox.zip`), reached the
goal in 539 of 540 validation trips with those walkers, with no collisions
([Fine-tuning in 3D](#fine-tuning-in-3d)). On 540 held-back trips in buildings and furniture never
trained on, the first model, the 2D policy, reached the goal in 536 (99.3%) with 1 collision, with the
rules in [Keeping going](#keeping-going-ask-walkers-go-round-or-wait).

## The worlds

| Building | Size | From |
| --- | --- | --- |
| `corridor` | 22 x 4.5 m | the 2D simulator's corridor room |
| `fp_hospital`, `fp_office`, `fp_warehouse` | 76 x 42, 31 x 95, 21 x 31 m | the 2D simulator's floor plans |
| `fp_depot`, `fp_warehouse3` | 30 x 15, 30 x 50 m | Nav2's depot and warehouse maps (Apache 2.0) |
| `fp_warehouse2`, `fp_house` | 14 x 21, 20 x 12 m | AWS RoboMaker's small warehouse and small house (MIT-0) |

Where each floor plan comes from and its licence are in [docs/data.md](../docs/data.md#sources).

`--variants 3` also makes `<name>_v1` to `_v3` of each, the same building with random furniture
(cabinets, shelves, benches and beds against walls, tables and crates in the open, always leaving
a 1.3 m passage), so training does not learn a few fixed layouts. The furniture is in the copy's
floor plan too, so routes go round it and touching it counts as hitting a wall.

The worlds are split so that testing uses only what training never saw (`splits.py`):

| | Worlds | Trips (seeds) |
| --- | --- | --- |
| Training | 7 buildings (all but the house) and their `_v1`-`_v3` copies: 28 worlds | 0-9999 |
| Test, new furniture | `<name>_test` of the same 7 buildings, a furniture layout not trained on | 10000-10199 |
| Test, new building | `fp_house` and `fp_house_test`, a building never trained on | 10000-10199 |

The test seeds 20000 and up were held back, as in 2D, and used once, for the held-back test of the
first model (`python webots/evaluate.py --split test`, see [docs/evals.md](../docs/evals.md#held-back-test-the-first-model)).

Every world has a rider sitting in the chair (`RIDER` in the world file). `bridge.py` seats them
again before every 20 ms physics step, from where the chair is and how fast it moves, so they stay in
their seat (within 0.3 cm). The chair's lidar and camera do not see them, and the first-person view
(**F**) is from their eyes, so nothing the chair senses changes: four hospital test trips (seeds 10000
to 10003) ended the same, with the same time and distance, with and without them.

## Use

| Task | Command |
| --- | --- |
| Watch the learned assistant drive | `python webots/run.py corridor` |
| Use a Webots you already opened on that world | add `--attach` (close other Webots windows on port 1234 first) |
| Take control | click the 3D view, press **O**, drive with the arrow keys, **O** again to hand back |
| Keys (click the 3D view first, and **L** lists them at the bottom of it) | see [Keys](#keys) below |
| Walk people round the chair from phones | `python webots/run.py fp_hospital --phones` (see [Phones](#phones)) |
| Compare drivers on the same people crossing the route | `python webots/crossing.py` (add `--world fp_warehouse2 --trips 10`) |
| Find rooms by a directory and signs, without the floor plan | `python webots/run.py fp_hospital --open-world` (see [Open world](#open-world-finding-rooms-by-the-directory-and-signs)) |
| Show the signs in an ordinary trip | add `--signs` to `run.py` |
| Measure sign reading and finding rooms | `python webots/sign_eval.py`, `python webots/open_eval.py` |
| Train a sign detector (`pip install ultralytics` first) | `python webots/sign_detector.py collect`, then `train`, then `evaluate` |
| Give commands by voice: hold **K**, say "stop", "go to Pharmacy" or "I need my prescription", let go | `bash setup.sh` once (it installs vosk and sounddevice and downloads the speech model). Voice is always on in `run.py` (see [Voice commands](#voice-commands)) |
| Find what makes a run slow on this computer | `python webots/profile_run.py fp_hospital --open-world` (any `run.py` options): the time each part of a step takes, saved to `data/traces/profile_run.txt`. Add `--drive` to keep the chair going, or `--override` to also take over for 20 s in every 30 |
| Start with another camera view | add `--view trip` (as the 2D demo's floor plans), `--view whole` or `--view first` (through the chair's eyes). The default follows the chair |
| Markers, pointer and sensor view: the rider picks where to go by looking | `python webots/run.py fp_hospital --landmarks --low-obstacles --sensors`, see [Pointing and markers](../docs/markers.md) |
| Show how the chair predicts people will walk, and what kind of walker each is | add `--tracks` (or press **T**), see [Predicted tracks and SiT data](../docs/people.md#predicted-tracks-and-sit-data) |
| Show the proxemic zones round the chair (intimate, personal, social, the front stretching with speed) | add `--zones` (or press **Z**), see [proxemics.md](../docs/proxemics.md) |
| Replace some walkers with recorded tracks (SiT) | `python scripts/sit_tracks.py path/to/SiT/labels`, then `python webots/run.py fp_hospital --replay data/sit/tracks.csv 6 --tracks` |
| A busier map, with people wherever the chair goes: new people appear out of its sight around it and along the route ahead, people far away (24 m, out of sight) disappear, and after **G** sends the chair somewhere new the new route fills up straight away | on by default in `run.py`, at the world's usual crowd; `--crowd-scale 1.5` for more, `--no-stream` to switch it off. Narrow places stay quiet (at most 0.05 people per square metre, about 3 in a 20 m corridor). The world has a fixed number of pedestrian models (22 to 42), which is the most people it can show at once |
| Another driver | `python webots/run.py fp_hospital --driver rule` (or `route_follower`) |
| Measure 20 trips, fast | `python webots/run.py fp_hospital --episodes 20 --fast` |
| Walls from the map instead of the lidar (as in 2D) | add `--walls map` |
| Hide the blue line the chair leaves behind | add `--no-trail` |
| Test the 2D policy on the 3D test set | `python webots/evaluate.py` (add `--parallel 4` to run 4 worlds at once) |
| Watch the latest model (fine-tuned with the SiT walkers and the proxemic reward) | `python webots/run.py fp_hospital --model models/sit_prox.zip --walkers learned` |
| Fine-tune the 2D policy in 3D | `python webots/train.py --envs 4 --timesteps 50000` |
| Test the fine-tuned policy | `python webots/evaluate.py --model models/assist_ppo_webots.zip` |
| Watch the fine-tuned policy | `python webots/run.py fp_house_test --model models/assist_ppo_webots.zip` |
| Cloud machine (Ubuntu, no screen) | `bash webots/cloud/setup_ubuntu.sh`, then `python webots/cloud/benchmark.py` to find how many copies of Webots the machine can run, then `python webots/train.py --envs N --timesteps 100000` (add `--headless` to `run.py`), see [docs/aws_training.md](../docs/aws_training.md) |
| Add the chair's speed and turn rate at every step to the results | add `--save-traces` to `evaluate.py` (saved to `data/traces/`) |
| Compare two runs on the same trips | `python eval/compare.py A.parquet B.parquet` (see [docs/evaluation.md](../docs/evaluation.md)) |
| Check that the chair stops short of walls while the rider drives at them | `python webots/override_check.py` (`--no-brake` for the same trips without the brake) |
| Count how often the chair turns on the spot, and why | `python webots/spin_check.py` (options at the top of the file) |

`run.py` opens Webots for you and closes it at the end. Every trip prints one line (success, hit a
person, hit a wall or timeout, with time and distance). With `--episodes` you also get a summary.

`evaluate.py` runs 20 trips in each of the 9 test worlds without a window, each trip three times
with three kinds of crowd:

- **spread**: people anywhere in the building, as on an ordinary day. This is the main score.
- **busy**: people placed within 6 m of the chair's route, as the 2D simulator does. This is a stress test.
- **events**: spread-out people plus situations (below). Calm crowds underneath, so a failure is down to the event.

It saves one row per trip to `data/episodes/<run_id>.parquet`, with the same columns as
`eval/run_episodes.py` plus `world_kind` (new furniture or new building) and `crowd`, and prints a
summary for each crowd. Crashes are split into `hit_moving` (the chair was driving) and
`hit_stopped` (it had stood still for half a second and someone walked or drove into it). Both
count as collisions. Every policy gets exactly the same trips (the chair is reset to exactly the
same state before each one), so two runs can be compared trip by trip. `--crowd spread` or
`--crowd busy` runs only one kind.

Every row also has comfort columns (stops per minute, hard brakes, speed near people, safety-stop
triggers, how far off the tracker's prediction was, and more), measured by `eval/comfort.py` and
defined in [docs/evaluation.md](../docs/evaluation.md). Success is close to 100%, so these are what tell
policies apart.

The people tracker only sees people in the chair's line of sight (a real chair cannot see through
walls). In 2D it sees everyone within 8 m. `--see-through-walls` gives the 2D behaviour.

### Phones

`--phones` lets up to 4 people walk round the chair from their phones, to test how the assistant
reacts. The laptop shows a QR code. Each phone, held sideways, shows what its person sees: the left
thumb walks, the right thumb looks round, and walking up to the chair the view looks down at it.
Walking into the chair counts as a hit and ends the trip, as any collision does. Phones and laptop
must be on the same Wi-Fi (university Wi-Fi often blocks this, so use a hotspot).

## People

By default, people walk by hand-written rules. With `--walkers learned` (run.py and evaluate.py),
ordinary people step the way Trajectron++, a published model trained on recordings of real
pedestrians, says they would. Their routes, how they react to the chair, and the walls still come
from the rules. See [walk_model/README.md](walk_model/README.md) for how it works, how to retrain
it, and the results.

In `run.py` the crowd follows the chair, so there are always people on the way ahead, and the test
set's events crowd adds the situations each building is known for, such as an emergency trolley in
the hospital or a forklift in the warehouses. `--tracks` (or **T**) shows where the chair predicts
each person will walk, and `--replay` swaps some walkers for real people recorded in the SiT dataset.
The crowd that follows the chair, every event and the tracks are in [docs/people.md](../docs/people.md).

## Fine-tuning in 3D

`train.py` fine-tunes a copy of the 2D policy with PPO in the 28 training worlds, one Webots per
environment, with random crowds, chair response, lidar noise and events in every trip. The first run
(50,000 steps, 2 November) made no real difference: one to three trips out of 140, within luck. A later
run on the AWS machine with the walkers trained on SiT and the proxemic reward is the latest model,
`models/sit_prox.zip`: 539 of 540 validation trips with those walkers, no collisions, and 9.6% fewer hard
brakes than the 2D policy. How it works, the results and its limits are in
[docs/webots_training.md](../docs/webots_training.md).

## Proxemic zones

`--zones` (or **Z**) draws personal space zones round the chair on the floor (`zoneview.py`): red
intimate, orange personal and yellow social, the front stretching up to twice as deep as the chair
speeds up. They follow Hall's proxemics, an idea from the team's mentor, Dr Francisco Cruz. Every trip
measures how long people spend in each zone, and `--reward proxemic` trains with a cost for driving into
them. The zones and the sensor view (**S**) are never shown together, as the two flicker where they
overlap. The design, the measures and the results are in [docs/proxemics.md](../docs/proxemics.md).

## Keeping going: ask walkers, go round, or wait

These eight rules around the policy stop the chair waiting for a gap that never comes
(`assist/controller.py`, `baselines/route_follower.py`):

1. **Excuse me, walkers only.** A walker who holds it up for 3 s within 60 cm in front is asked to
   step aside. People standing to talk or wait are never asked: the route goes round them.
2. **Round a blocker.** Held up 4 s by someone in front, it plans round them for 10 s.
3. **Keep to one way.** It keeps its route unless someone has stood on it for 5 s or a new route is
   3 m shorter.
4. **Let faster walkers pass.** It slows to 15% for someone overtaking within 30 cm.
5. **Wait for people cutting across**, or walking in towards its front.
6. **Half speed** with someone walking within 50 cm.
7. **Get out of a crush or a wall corner** after 2 s, by a way ahead if there is one.
8. **Do not turn round on the spot.** A new route back the way it came is taken only after 8 s of
   waiting for the way on. After turning back it keeps that way for 10 s, and getting out of a crush
   keeps close to the route (`spin_check.py` counts the turning that is left).

| 540 validation trips (seeds 10000 to 10019), the learned 2D policy | before the rules (10 Nov) | 7 rules (10 Nov) | 8 rules, wall brake, rider (18 Nov) |
| --- | --- | --- | --- |
| Reached the goal | 533 (98.7%) | 537 (99.4%) | 537 (99.4%) |
| Timeouts | 6 | 3 | 3 |
| Crashes, chair moving / standing | 1 / 0 | 0 / 0 | 0 / 0 |

The rules change the test a little: walkers can be asked, people standing still always leave the
chair a way, and people stop 2 cm short of it. The rules in full, comfort numbers and how much the
results move between reruns are in [docs/webots_results.md](../docs/webots_results.md). Teaching
the policy to do this itself is left for future work.

### When the rider drives

While the rider drives (**O**), the chair still watches for people straight ahead (`guard.py`): the
ones its sensors can see, as the green rings show. Within 1.5 m, a red warning takes the top line
("PERSON 0.8 m AHEAD: slowing down") and a red ring marks the person on the floor. The chair slows
down too: full speed until 1.2 m, then less and less, and no forward speed at 0.3 m. Turning is never
held back, so the rider can always turn away. `--no-brake` keeps the warning without slowing down.
People hit while the rider drives are counted on the top line. What the sensors see (**S**) comes on
by itself while the rider drives and goes back as it was after, and **S** still hides it for that time.

The chair does the same for walls and anything else the lidar sees straight ahead, such as furniture, a
pallet or a low table. The gap is measured from the chair's front edge, within its width, so the walls
of a corridor beside it and the sides of a 0.9 m doorway do not count. Within 0.8 m the top line says
"OBSTACLE 0.6 m AHEAD: slowing down", the chair slows down from 0.8 m and has no forward speed at
0.12 m ("stopped: turn away"), and turning is never held back. A person ahead is warned about first.
Walls hit while the rider drives are counted next to the people hit, by the trip's rule, which misses
a chair that only touches a wall ([known gaps](../docs/evals.md#known-gaps)).

Driving straight at the nearest person in 4 take-overs of 20 s, the chair hit 3 people with
`--no-brake` and none with the brake (`python webots/profile_run.py fp_hospital --open-world --steps
1200 --at-people`, with `--no-brake` after the world for the first).

Driving at a wall 1 to 3 m ahead and holding forward for 6 s, in the 8 of 10 hospital trips that
reached a wall, the chair stopped with 0.06 to 0.11 m to spare every time, and touched the wall in 5
of 8 with `--no-brake` (`python webots/override_check.py`).

## Keys

Click the 3D view first. **L** shows this list at the bottom of it.

| Key | What it does |
| --- | --- |
| **O** | Take control of the chair, or hand back to the assistant. With `--landmarks`, taking control hangs 6 new markers around the chair |
| **Arrows** | While you have control: **Up** forward, **Left** and **Right** turn, no key stops. Turning also sweeps the red aiming line |
| **1** to **8** | Pick the marker with that number (manual mode), and a white beam stands on it. **0** clears the pick |
| **G** | The assistant drives to the locked or picked marker, or to the red post if none is locked. The trip ends when the chair arrives |
| **S** | Show or hide what the chair's sensors can see (`--sensors` starts with it on). It hides the proxemic zones, as the two flicker where they overlap |
| **T** | Show or hide each person's predicted track (orange), walked track (blue), real track when replaying (black) and a ring coloured by kind of walker (`--tracks` starts with it on) |
| **Z** | Show or hide the proxemic zones round the chair: red intimate, orange personal, yellow social, the front stretching as the chair speeds up (`--zones` starts with them on). It hides the sensor view, which then does not come on by itself while the rider drives |
| **F** | First-person view: the rider's eye (1.15 m up), then each phone walker's, then the overview |
| **M** (or **V**) | Camera: around the chair, the trip (as the 2D demo), the whole building |
| **Space** | Pause and resume (the trip clock stops too) |
| **L** | Show or hide the key list |
| **H** | Hide or show the status lines and lists at the top, for a clear first-person view |
| **R** | Restart the trip with a new seed (`--open-world`: the same start and people) |
| **K** | Hold it, say what to do, let go (see [Voice commands](#voice-commands)) |
| **N** | Open the next building in a new Webots |
| **Q** | Quit |

With `--open-world`, **1** to **9**, **P**, **D**, **G** and **C** pick from the directory (see
[Open world](#open-world-finding-rooms-by-the-directory-and-signs)). While you drive (**O**), the number
keys and **G** are the markers' again.

## Pointing and markers

A rider in a building they do not know can still say where to go, by looking at it. With
`--landmarks`, every take-over (**O**) hangs 6 numbered markers, each its own colour and shape, on the
farthest walls in sight. A red line runs from the chair along its heading and locks on to a marker as
the rider turns, and **G** sends the chair there, round walls, furniture and people. Low tables
(`--low-obstacles`, on by default) are seen over by the rider and driven round by the chair, and
`--sensors` (**S**) shows what its sensors see. How it works and its limits are in
[docs/markers.md](../docs/markers.md).

## Open world: finding rooms by the directory and signs

This finds a named room without the floor plan, as a visitor would: read a directory on the wall,
then follow the signs (`signs.py`, `vision.py`, `explore.py`, `open_world.py`). Everything else is an
ordinary trip, and every `run.py` option works with it.

```
python webots/run.py fp_hospital --open-world                      # G goes to the directory, then you pick
python webots/run.py fp_hospital --open-world --target "CT Scan"   # straight to one place
python webots/run.py fp_hospital --open-world --landmarks 0        # without the markers and red line
```

A trip has three steps:

1. **Read the directory.** **G** drives to the nearest one and holds still until it is read.
2. **Pick a department** (**1**-**9**, then **G**). The chair follows the wing and department signs.
3. **Pick a room** from the places read nearby, then **G**.

Each sign carries a tag (ArUco) that places it within about 0.1 m, and its words are read with OCR
within about 5 m. In 10 hospital trips the chair reached every place, taking about 6 times as long
as with the floor plan, mostly finding a directory. If it is stuck, walkers step aside when asked,
and the rider can take control (**O**) and drive past. How it works and all the results are in
[docs/webots_open_world.md](../docs/webots_open_world.md).

Keys: **1**-**9** pick from the list, **P** next page, **D** switch between the directory and the
places read nearby, **G** go, **C** stop. Markers are put up along the walls of the whole building
at the start and stay for the whole trip, whoever drives: one about every 6 m, each its own colour
and shape pair out of 8 colours and 6 shapes (33 in the hospital), none within 1.5 m of a sign. From
anywhere in the hospital 4.3 are in sight on average, and never none (80 spots). The nearest ones in
sight, up to 9, are numbered from left to right and are the ones the rider can send the chair to:
by voice at any moment ("go to the red pyramid", or "the red one" when only one red is in sight),
and while the rider drives (**O**) with the red aiming line, the number keys and **G**, as with
`--landmarks`. The markers and the red line are on the screen only: the chair's camera does not see
them. A green ball hangs over every sign the chair has read.

Long trips do not slow down: see [speed on long trips](../docs/webots_open_world.md#speed-on-long-trips).

## Voice commands

The rider can say what to do instead of pressing a key (`voice.py`). Exact commands are recognised
on the laptop, offline. Anything else, in the rider's own words ("I need my prescription"), is worked
out by a language model on the course gateway (`interpret.py`), and the chair asks before it moves.

```
bash setup.sh                           # once: vosk, sounddevice and the speech model (40 MB, models/)
python webots/voice.py --mic            # say a command: how loud the microphone heard it and what was recognised
python webots/run.py fp_hospital --open-world   # voice is always on: hold K and speak
```

Own words need `COMP3110_MODEL_BASE_URL`, `COMP3110_API_KEY` and `COMP3110_MODEL_NAME` in `G19/.env`
(kept out of Git). `bash setup.sh` makes the file with the course's address and the model filled in
and the key empty: put your own key in. Without a key, or without an answer within 3 s, only exact
commands work.

Click the 3D view, hold **K**, speak and let go. The last status line says what was heard and what
the chair does, and the Terminal shows how long and how loud each recording was. Each command comes
in as the key it stands for, so the chair drives exactly as it does with the keys.

| Say | What it does | Key |
| --- | --- | --- |
| "stop", "wait", "hold on" (anywhere in what is said) | the rider takes control, so the chair stops at once | **O** |
| "take control", "let me drive" | the same | **O** |
| "hand back", "you drive" | the assistant drives again | **O** |
| "go ahead", "carry on" | go to the place or marker picked (open world, nothing read yet: to the directory) | **G** |
| "go to Pharmacy", "take me to CT Scan" | open world: a place the chair has read, in the directory or nearby | pick, **G** |
| "go to the yellow sphere", "go to the yellow one" | a marker on the wall in sight (`--landmarks`, and always up in the open world). "The yellow one" works when only one yellow is in sight | pick, **G** |
| "go to number three" | number 3 on the list on the screen (while driving with markers: marker 3) | **3**, **G** |
| "next page" | the next page of the open world's list | **P** |
| anything else ("I could do with a coffee") | the model picks a command from what is on the screen, and the chair asks first: "Cafe? Say yes or press G" | pick, then **G** or "yes" |

How a recording is recognised, how the model was chosen and how well both work are in
[docs/voice.md](../docs/voice.md). In short, on 114 test requests in riders' own words the chosen model
was right on 92% with no wrong moves, and exact commands read by 6 computer voices were understood 88%
of the time in quiet and 63% in loud noise, with none of the 1,680 sending the chair somewhere it was
not asked to go.

## Files

```
make_world.py      floor plan -> worlds/<name>.wbt (3D walls, furniture) + <name>.json (which plan, how many people)
splits.py          which worlds are for training and which for testing
protos/Wheelchair.proto   the chair: two drive wheels, two casters, lidar, camera
bridge.py          makes Webots look like the 2D environment to assist/, baselines/ and training/, seats the rider
people.py          the walking people: envs/pedestrians.py's rules, drawn as Webots pedestrians
learned_walk.py    --walkers learned: people step as Trajectron++, trained on real pedestrians, suggests
walk_model/        get, train, export and test that model (walk_model/README.md has how and the results)
run.py             watch, take control, or measure trips, the keys, pointer and marker handling, --open-world
landmarks.py       coloured wall markers, the selector beam and the red aiming line
obstacles.py       low obstacles (tables) in the world and the floor plan
sensorview.py      the sensor fan and the rings round people the chair sees
zoneview.py        --zones: the proxemic zones round the chair, drawn on the floor
guard.py           while the rider drives: the warning and slowing down for people, walls and furniture in the way
override_check.py  the rider drives at walls: does the chair stop short (with and without the brake)
spin_check.py      how often the chair turns on the spot, and which part of the code was in charge
trackview.py       each person's predicted, walked and real track, their kind of walker, and the prediction score (ADE, FDE)
replay.py          recorded tracks (SiT, via scripts/sit_tracks.py) walked by people in the simulator
players.py         who walks whom from the phones, and their coloured rings
phones.py          --phones: the web server, the page and each walker's 3D view
crossing.py        the same people crossing the chair's path for every driver
signs.py           door plates, blade and direction signs, directories, with their tags
vision.py          reading signs with the chair's camera
explore.py         the chair's own map from its lidar, and following direction signs
open_world.py      run.py --open-world: read a directory, then follow the signs
sign_eval.py       how well signs are read, by distance and angle
open_eval.py       finding a named room with and without the floor plan
sign_detector.py   a YOLO sign detector trained on pictures that label themselves
voice.py           always on in run.py: hold K and speak, recognised offline (Vosk) as the keys it stands for
interpret.py       the rider's own words to one command, by a model on the course gateway
voice_eval.py      how well exact voice commands are understood: recorded or computer voices, in noise
interpret_eval.py  which gateway model works out own words best (voice_requests.jsonl: 231 requests)
get_voice_model.sh the speech model for voice.py, into models/ (setup.sh runs it)
profile_run.py     where the time of each run.py step goes, on this computer
webots_env.py      training/assist_env.py's AssistEnv, with trips in Webots
train.py           PPO fine-tuning, one Webots per CPU core
evaluate.py        the 3D test set: many trips without a window, results saved as in eval/
launch.py          finds and starts Webots, connects our Python to it
install_webots.sh  installs Webots R2025a and checks that the Python environment can reach it
cloud/setup_ubuntu.sh   one-step setup of an Ubuntu 24.04 cloud machine (not yet tried on it)
cloud/benchmark.py      how many copies of Webots one machine should run (steps per second for each number)
```

## What is the same as 2D, and what is not

| | 2D | Webots |
| --- | --- | --- |
| Building, start, goal, people's rules | floor plan | the same floor plan, walls 2.5 m high |
| Chair's movement | moved directly | two motor-driven wheels in the middle (turns on the spot) and physics |
| Wall distances (16 directions) | from the floor plan | from the lidar, which also sees people and plants (`--walls map` for the 2D way) |
| People | our rules (`envs/pedestrians.py`) inside SocNavGym | the same rules, or learned walkers (`--walkers learned`), placed each step (not pushed by physics) and animated |
| People tracking | true positions + 3 cm noise, everyone within 8 m | the same, but only people in line of sight |
| Where the chair is | known exactly | known exactly (a real chair would have to estimate it) |
| Collisions | centre within 0.4 m of a wall or 0.76 m of a person | the same rule |
| Time step | 0.1 s | 0.1 s per decision, physics every 20 ms |
