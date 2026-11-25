# People in the 3D simulator

The crowd that follows the chair in `run.py`, the events in the test set, and each person's predicted
and recorded tracks. The short version is in [webots/README.md](../webots/README.md#people). The
learned walkers have their own page: [webots/walk_model/README.md](../webots/walk_model/README.md).

## Crowd that follows the chair

People used to be placed once, near the first route, so after **G** (a new goal) or a long drive
there was nobody ahead. With streaming (`Crowd.stream` in `people.py`, switched on in `run.py`) the
crowd follows the chair: every second the walkers around the chair (within 14 m) and along the next
25 m of route are counted, and if there are fewer than the world's usual crowd times `--crowd-scale`
(and at most 0.05 people per square metre, so a corridor holds only a few), new people or groups
appear at least 8 m away and out of the chair's sight (or further than 12 to 16 m), 60% of them on
the route ahead.

New people walk *through*: each heads for a spot at least 12 m away and out of the chair's sight and
is taken away when they get there (with their group). Everyone else who arrives somewhere walks on,
ahead of the way they were going, to somewhere at least 6 m away (within 60 degrees of their
heading), so nobody paces up and down a corridor. Walkers more than 24 m away and out of sight (35 m
in sight) are also taken away, which frees their pedestrian model. Pressing **G** calls
`scene.set_goal`, which plans the new route and fills it at once. Training and testing (`train.py`,
`evaluate.py`) leave streaming off, so their trips stay the same as before; the walk-on rule applies
everywhere, so results from before it are not comparable. (The `two_way_traffic` event of the events
crowd still walks people up and down the chair's route on purpose.)

## Events

| Where | Event | What happens |
| --- | --- | --- |
| Everywhere | Older people | 0.4-0.7 m/s, fewer notice the chair, stop for 1.5-4 s now and then |
| Everywhere | Someone stepping out of a room | waits out of sight near the route, walks across it as the chair comes |
| Corridor | Two-way traffic | most people walk up and down along the chair's route, both ways |
| Hospital | Waiting area | 3-6 people standing together 2-5 m from the route |
| Hospital | Slow patients | 0.3-0.5 m/s and 0.9 m wide (drip stand or walking frame; drawn as a person) |
| Hospital | Emergency trolley | a 2 m trolley and two staff rushing (1.5-2 m/s) towards the chair, not making way |
| Office | Meeting ends | 4-8 people leave one room together |
| Office | Person on the phone | slow, not watching, drifting from side to side |
| Warehouses, depot | Box carriers | do not see the chair in time |
| Warehouses, depot | Pallet of boxes | stands beside the route, not on the map |
| Warehouses, depot | Forklift | 2.5 m long, 2-2.5 m/s, towards the chair along its route, never yields |
| House (test only) | Child, pet | small and fast, change direction every 1-3 s |

Objects (trolley, forklift, pallet, child, pet) count as circles for collisions (trolley 1.4 m,
forklift 1.8 m across) and are drawn as boxes of their real size. The people tracker reports them
like people, as a camera-based detector would. All the numbers are at the top of `people.py` and in
`BUILDING_EVENTS` in `bridge.py`.

## Predicted tracks and SiT data

`--tracks` (or **T**) draws, for each of the 10 nearest people the chair's tracker follows
(`trackview.py`): an **orange** ribbon, the track the chair predicts for the next 3 s (it assumes they
keep walking as they have over the last second); a **blue** ribbon, where they walked in the last 3 s;
and a ring on the floor coloured by what they are like: **green** walking, **magenta** on the phone,
**blue** in a group, **yellow** stopped or waiting, **purple** older, **white** standing, **orange** a
child, pet or trolley, **black** someone walking a recorded SiT track (below). It also scores the prediction as SiT scores trajectory prediction: ADE (the
mean error of the predicted positions 0.5 to 3 s ahead) and FDE (the error at 3 s), shown in the
last-trip row and printed at the end of each trip.

[SiT](https://spalaboratory.github.io/SiT/) (Bae et al., NeurIPS 2023) is a dataset of pedestrian
tracks recorded from a robot driving through crowds, indoors and outdoors, at 10 Hz. Access is by a
Google Form on its project page, so the data is not in this repository (`data/sit/` is ignored; its
licence is share-alike on the project page and no-derivatives on its GitHub, so do not republish it).
With the data downloaded (`python scripts/download_sit.py DEST` fetches it from the public Drive
folder and resumes; the zips are 3 to 7.5 GB each, about 300 GB in all, so use a big disk, and Google
may refuse anonymous downloads of popular files for up to 24 hours, in which case download them in a
signed-in browser instead):

1. `python scripts/sit_tracks.py path/to/SiT --out data/sit/tracks.csv` (the folder of scene zips, or of
   unpacked label files) turns its 3D pedestrian labels into one table of `id,t,x,y` and prints the
   median walking speed. Seated people (`Pedestrain_sitting`) are left out. The labels are already in a
   fixed frame (seated people do not move in them), so no pose correction is needed for SiT.
2. `python scripts/fit_behaviour.py data/sit/tracks.csv --format table --columns id,t,x,y` fits walking
   speed, stops and groups, to paste into `envs/configs/base.yaml` (`project.behaviour`), so the
   crowd walks like real people.
3. `python webots/run.py fp_hospital --replay data/sit/tracks.csv 6 --tracks` puts 6 recorded people in
   each trip (`replay.py`): ordinary walkers are taken over, and where a building has fewer walkers than
   that (the hospital often has 2 or 3) new ones are added in the world's spare pedestrian models. Each
   follows one real track step for step, with that person's speed, stops and turns, moved onto a free
   spot at least 3 m from the chair and cut short where it would meet a wall; afterwards they walk on as
   ordinary walkers. A **black** ring marks them, and a thin **black** line on top of the orange ribbon
   shows where they really walk next 3 s, beside what the chair predicts. The recording has no chair or
   other people in it, so a replayed person does not steer round them, but they do not walk into them
   either: they stop when the next step would take them within 15 cm of anyone, and the chair, and the
   recording waits with them. After 4 s held up they give up the recording and walk on as an ordinary
   walker, and they never move faster than 2 m/s catching up.

`scripts/sit_tracks.py` was checked against SiT's Corridor_1 (13 walkers, median 0.8 m/s, seated people
still). `tests/test_sit_tracks.py` uses made-up label files and tracks, not SiT data.
