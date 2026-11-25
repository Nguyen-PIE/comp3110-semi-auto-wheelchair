# Open world: finding rooms by the directory and signs

How `run.py --open-world` finds a named room without the floor plan, and how well it does. The short
version and the commands are in [webots/README.md](../webots/README.md#open-world-finding-rooms-by-the-directory-and-signs).

## How it works

A real chair does not get the building's floor plan or a list of its rooms. With `--open-world`,
`run.py` takes both away: the chair reads a directory on the wall, as a visitor would, then finds its
way by the signs. Everything else is an ordinary trip: the same people (streaming, events), the same
chair and 4 m lidar, the learned assistant (RL) driving, and every option (`--phones`, `--landmarks`,
`--low-obstacles`, `--sensors`, `--crowd-scale`, `--driver`, `--model`, `--view`). Code: `signs.py`,
`vision.py`, `explore.py`, `open_world.py`.

```
python webots/run.py fp_hospital --open-world                      # stands still: G goes to the directory, you pick
python webots/run.py fp_hospital --open-world --target "CT Scan"   # straight away, as an appointment letter says
python webots/run.py fp_hospital --open-world --phones             # with people walked from phones
python webots/run.py fp_hospital --open-world --landmarks 0        # without the markers and red line
python webots/open_world.py fp_hospital                            # the same as run.py ... --open-world
```

1. **Wings and signs** (`signs.py`). Hospitals are split into wings or zones, each with a colour
   (Sunshine Coast University Hospital: Zone A blue, Zone B green, Emergency red), with a directory
   near the main entrance and at the lifts (UC Davis Health). Here the hospital has a West, Central and
   East Wing, the office a North, Central and South Wing. Four kinds of sign:
   - a door plate beside each door, 1.45 m up (the NCC puts tactile signs 1.2 to 1.6 m up)
   - a blade sign above it, 2.15 m up, read from down the corridor
   - a direction sign at each corridor junction, 2.25 m up: the other wings first, then this wing's
     departments, four rows at most (NHS guidance: four or five destinations per sign)
   - a directory on the wall, 1.40 m up, 1.6 x 1.0 m: "You are here" and every department and public
     place with its wing, Emergency first, then A to Z. One hangs at the main reception, then more until
     every junction is within 22 m of one (the hospital has 4, the office 5).

   A department with several rooms (a ward) also gets a sign at its entrance. Every room in every
   building is named: by hand in the hospital, office, corridor and house, from lists in the warehouses
   and depot (whose directory hangs at Goods In). The house has door signs only (as dementia-friendly
   homes use) and no directory. A building's variants (`_v1`, `_v2`, `_v3`, `_test`) have the same
   names. Signs are added while Webots runs and have no physics, so training and testing are unchanged.
2. **Reading them** (`vision.py`): the chair's camera (960 x 540) is read 5 times a second. Each sign
   carries a small tag (ArUco, as some hospitals put QR codes on signs), which gives where the sign is
   and which way it faces. The words are read with OCR (RapidOCR) off the sign, straightened out from
   the tag, and arrows by their shape. A name counts once two readings agree, and a reading that is only
   the start of a name ("4B Room" for "4B Room 4", "Bed" for "Bedroom") does not count. Where a door's
   plate and blade sign disagree, the name read most often wins. A directory is read row by row, each
   row counting once two readings agree, and a wing misread at an angle ("Eastwing") is put right.
   `--read tags` takes names from the tags instead (no OCR).
3. **Its own map** (`explore.py`): from the same 4 m lidar the assistant uses, with unknown space
   counted as open, planned again as walls appear. Floor counts as explored once the chair has been
   within 6 m of it, since the camera cannot read words from further off. Where the chair is comes
   from the simulator: a real chair would need SLAM, which this project does not do.
4. **A trip**, in three steps (`open_world.py`):
   1. *Read the directory.* The chair stands still, with the shortest path to the nearest directory
      drawn on the floor in orange (from the floor plan, as a "You are here" map shows the way). **G**
      drives that path and holds still 2.1 m in front of the board until it is read whole (or the
      rider drives there, **O**). If someone stands in front, it tries from one side, then the other.
      A home has no directory: **G** explores, reads the room signs and the rider picks from the
      places read. (`open_eval.py` has the chair find a directory itself: it looks round, explores,
      follows signs for the Main Reception and goes up to any directory it spots.)
   2. *Pick a department* (**1**-**9**, then **G**). The directory is read once, as a visitor notes
      the wing: the chair passes other directories by. It follows the signs for the wing, then for
      the department, goes up to signs it has spotted but not read, and keeps near the department's
      rooms once it has read some. It has arrived when it reads the department's own sign or one of
      its rooms. The goal post only stands on a real destination (the directory, or a place whose
      sign it has read), never on the spots it explores.
   3. *Pick a room.* The list switches to the places read within 15 m. Pick one and **G** again
      (Medical Imaging, then CT Scan).

   Names match as a visitor would: a misread letter is allowed, a different number is not ("4B Room 1"
   is not "4B Room 5"), and names are compared word by word ("Imaging Reception" is not "Main
   Reception"). If its map closes the gap it came in through (between furniture, or someone in a
   doorway), or it has squeezed into a gap its map then shows is too narrow, the chair backs out along
   its own trail. Where it got stuck it tries again a minute later, since people standing in a
   corridor are walls to it until they move. Walkers in its way step aside when it asks, as in every
   trip (see [Keeping going](../webots/README.md#keeping-going-ask-walkers-go-round-or-wait)). If they cannot, or
   people stand there to talk, the screen says so and the rider can take control (**O**) and drive past. A trip does not end when the chair arrives or
   someone bumps it: **R** starts again from the same start with the same people, knowing nothing,
   and **N** opens the next building.

Keys, besides `run.py`'s **O**, arrows, **Space**, **M**, **F**, **S**, **L**, **H**, **R**, **N**, **Q**:
**1**-**9** pick from the list on the screen, **P** next page, **D** switch the list between the
directory and the places read nearby, **G** go (before a directory is read: to the directory), **C** stop.
Taking control (**O**) puts up new markers and the red aiming line, as `--landmarks` does in an
ordinary trip: the number keys and **G** then pick a marker and send the chair there (or to the red
post). They show only while the rider drives, and come down on **G** or handing back. The chair drives
to the marker on its own map, reading signs on the way, and waits there. **G** in autonomous mode goes
back to the directory list. `--landmarks 0` switches the markers off. A green ball hangs over every sign the chair has
read, where its camera put it.

## Results

**Reading signs** (`sign_eval.py`, the hospital: 1233 views of its 160 sign faces, 1.5 to 12 m away and
up to 70 degrees to the side, one picture each, read once, nobody in the way):

| | views | tag found | position error (median, 90%) | words read exactly |
|---|---|---|---|---|
| all | 1233 | 71% | 0.09 m, 0.33 m | 36% |
| door plates | 360 | 78% | 0.06 m, 0.25 m | 45% |
| blade signs | 705 | 69% | 0.13 m, 0.37 m | 34% |
| direction signs (every row and arrow) | 136 | 63% | 0.10 m, 0.33 m | 29% |
| directories (every entry, wing and "You are here") | 32 | 78% | 0.08 m, 0.18 m | 12% |
| 1.5 to 3 m | 491 | 66% | 0.04 m, 0.06 m | 48% |
| 3 to 5 m | 363 | 88% | 0.11 m, 0.19 m | 58% |
| 5 to 7 m | 164 | 76% | 0.24 m, 0.33 m | 1% |
| 7 to 9 m | 99 | 63% | 0.35 m, 0.50 m | 0% |
| 9 to 12 m | 116 | 47% | 0.47 m, 0.71 m | 0% |

The tag puts a sign within about 0.1 m and gives the way it faces to within a degree (median), up to
12 m away. Words are read only within about 5 m, so the chair reads a name as it comes nearer, and goes
up to signs it has spotted but not read. Close up, signs are often partly out of the picture. A name
counts once two readings agree, so one bad picture does not name a place. A directory is rarely read
whole from one picture taken anywhere, so the chair goes up to it and holds still: from its look spots
(1.7 and 2.1 m in front) every one of the 59 directories in the 35 buildings that have one was read
whole, every entry with its wing, from a single picture each. Tags elsewhere in the picture were given
the wrong number, or put more than 1.5 m off, 183 times in 1233 views: a name read off the sign itself
is unaffected, and a reading from more than 14 m is not used.

**A detector instead of tags** (`sign_detector.py`): YOLO11n fine-tuned for 15 epochs (CPU) on 369
pictures from the hospital, office, warehouse3 and corridor (1398 signs, 70 of them directories).
Share of the signs in view that it finds, against the tag reader:

| | detector | tags |
|---|---|---|
| new pictures of those buildings (480 signs) | 91% (precision 81%) | 49% |
| the depot, never trained on (176 signs) | 90% (precision 79%) | 45% |
| 10 to 14 m (both sets) | 87% | 24% |
| 14 to 30 m (both sets) | 73% | 5% |

Beyond 10 m it finds 3.6 to 15 times as many signs as the tags do, and needs nothing stuck on the signs,
but it does not say which sign it is or where it is: the chair would still have to drive up to read it.
`open_world.py` does not use it.

**Finding a place without the floor plan** (`open_eval.py`: the hospital, 10 trips from where `run.py`'s
trips with seeds 10000 to 10009 start, each to a place the directory lists, picked at random at least
10 m away, people around, 400 s limit with finding the directory included, the learned assistant
driving both ways, run in two halves of 5):

| destination | with the floor plan | without it, by the directory and signs | directory read after | places read |
|---|---|---|---|---|
| Waiting Area | 45 s, 37 m | 242 s, 192 m | 60 s | 29 |
| Main Reception | 52 s, 43 m | 355 s, 237 m | 24 s | 26 |
| Allied Health | 41 s, 42 m | 178 s, 142 m | 150 s | 21 |
| Emergency | 113 s, 94 m | 282 s, 234 m | 15 s | 31 |
| Allied Health | 26 s, 18 m | 170 s, 132 m | 133 s | 21 |
| Waiting Area | 32 s, 27 m | 330 s, 232 m | none in 4 min | 28 |
| Toilets | 34 s, 25 m | 296 s, 254 m | none in 4 min | 40 |
| Cafe | 17 s, 19 m | 311 s, 247 m | none in 4 min | 34 |
| Main Reception | 15 s, 13 m | 40 s, 20 m | 38 s | 2 |
| Pathology Collection | 39 s, 34 m | 111 s, 90 m | 76 s | 14 |

Both ways got there every time, without the floor plan stopping within 0.7 m of the true spot (1.7 m
once). It read a directory in 7 of 10 trips (median 60 s). In the other 3 it stopped looking after 4
minutes, and the place was then picked from the places it had read, which it reached by its signs. A
trip without the floor plan took 2.5 to 18 times as long (median 6), mostly finding a directory: the
hospital has 4, and people standing in front of one, or in a narrow corridor, keep the chair away for a
while. Collisions: none with the floor plan, and 0.6 s of contact in one trip without it (the Main
Reception trip, before open_eval said what each collision was).
Ten trips is a rough guide: people walk differently in each run, which changes the way the chair
explores first.

**Where the time goes once the place is picked** (`open_eval.py --ways signs --given-directory --ocr-s
0.2 --trace`: the directory known at the start, and signs read as late as `run.py`'s reading thread
reads them, 0.2 s a sign, 6 hospital trips). It got there in 5 of 6 (not the Main Reception), taking
61 to 285 s (median 244 s). Of that time 88% was spent driving and exploring the corridors and 12%
going up to signs and holding still in front of them (direction signs 5%, door signs 7%, 2 to 10 s
a look). It never went inside a room. So a faster search needs better choices of where to explore
next, more than fewer looks. A version with fewer door looks, slowing down to read doors in passing
and keeping out of the doorways of rooms already read got there in only 2 of 5 trips, and was dropped.

## Speed on long trips

A trip left running does not slow down. Two things would grow with it: working out where each sign
read is from all its sightings, several times a step, and adding to one blue line that Webots redraws
whole each time. Sign positions are worked out again only when a sign is seen again, and the line is
drawn in 5 m pieces. Over 3000 steps on a computer without a graphics card, a step went from 150 ms
to 370 ms before and stayed at 170 to 200 ms after.

A moving chair also cost more each step than a standing one, most of all while the rider drives.
Adding a point to the blue line, and changing a line of text at the top (5 labels with its white
outline), each made Python wait for Webots, and while the chair moves both happen every step. A new
piece of the line comes with all its points, so a new point only moves one, and text that changes
all the time (the speed, the distance to go) is sent at most 4 times a second. Switching between the
old and the new way every 20 s in one run here, with the rider taking over for 20 s in every 30, a
step took 791 ms the old way and 657 ms the new way (17% less) on a computer without a graphics card
running other simulations. In 2400 steps each way, the line went from 30 ms a step to under 1 ms and
the text from 35 ms to 8 ms. `python webots/profile_run.py fp_hospital --open-world --steps 6000 --every 500 --override`
shows it on any computer: `--drive` keeps the chair going from place to place, as a rider would with
1-9 and **G**, and `--override` also takes over (**O**) for 20 s in every 30 and drives with the
arrow keys.

## References

- **ArUco tags.** Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J. and Marín-Jiménez, M. J. *Automatic Generation and Detection of Highly Reliable Fiducial Markers under Occlusion.* Pattern Recognition 47(6), 2280-2292, 2014. Read with OpenCV's `cv2.aruco`.
- **OpenCV.** Bradski, G. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools, 2000. [opencv.org](https://opencv.org).
- **RapidOCR** reads the sign text, [github.com/RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR).
- **YOLO11n**, the sign detector. Jocher, G. and Qiu, J. *Ultralytics YOLO11*, version 11.0.0, 2024. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). Licence: AGPL-3.0.
- **Hospital signs.** NHS Estates. *Wayfinding: Effective Wayfinding and Signing Systems, Guidance for Healthcare Facilities.* 2nd edition, The Stationery Office, 2005. [PDF](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/148500/Wayfinding.pdf).
