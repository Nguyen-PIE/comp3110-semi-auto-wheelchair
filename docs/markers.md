# Pointing and markers (dynamic navigation)

The short version is in [webots/README.md](../webots/README.md#pointing-and-markers).

The idea: a rider in a building they do not know can still say where to go, by looking at it. The
chair does the routing, round walls, furniture and people. Code: `landmarks.py`, `obstacles.py`,
`sensorview.py`; the keys are handled in `run.py`.

1. **Markers on walls.** Every time the rider takes control (**O**), and at the start of each trip,
   the old markers come down and 6 new ones go up. The rider's view all round is cut into 6 equal
   sectors, the first centred on straight ahead; in each, a marker goes on the farthest wall in sight
   (at most `AIM_RANGE_M`, 30 m, as a real chair's sensors). So the same spot and heading always give the same
   places, and the farthest wall anywhere is always marked. They hang at head height with a disc on
   the floor, numbered from the chair's left to its right, each with a unique colour and shape ("yellow
   sphere"). Only places the chair could drive to are marked.
2. **Pointer.** In manual mode a thin red line runs from the chair along its heading to the first wall
   (30 m at most), with a red post where it ends. A marker within about 5 degrees of the line (or within
   half a metre of it, for a near one) locks on: the white beam stands on it and the bottom row names it.
   Turning the chair sweeps the line, so markers lock on as they come into line. Keys **1** to **8**
   pick by number instead.
3. **Go.** **G** hands over to the assistant, which plans a route to the spot in front of the locked
   marker (or the red post) and drives it with its usual people-avoidance. Arriving ends the trip.
   Press **O** to take control again and new markers appear from where the chair now is, so the rider
   crosses a building in hops.
4. **Look over, drive round.** With `--low-obstacles` (on by default; `--low-obstacles 0` turns it
   off), 4 obstacles stand on the floor each trip, two of them large dining or meeting tables (75 cm)
   and the rest coffee tables, poufs and planter boxes (about 40 cm). A seated rider's eyes (1.15 m) see over them, so the red line and the markers go
   past; the chair's lidar (33 cm) hits them, so the route goes round, and touching one counts as a
   wall collision. They are stamped into the floor plan (`plan.low` marks them) and are new every trip.
5. **What the sensors see.** `--sensors` (or **S**) draws a pale fan on the floor out to where each
   lidar ray stops, so a table cuts a notch in it, and a green ring under each person the people
   tracker can see (line of sight only; a table blocks it, as it blocks the lidar).

Limits: the sim knows the whole floor plan, so routing to anything in sight always works (a real chair
in an unmapped building would have to explore); the pointing is perfect (real gaze or head pointing
is about 0.5 to 1 degree off, which is 0.3 to 0.5 m at 30 m, so a real chair would snap to a labelled
marker as here); the markers are coloured shapes, not doors; and a marker can be named by voice
([Voice commands](../webots/README.md#voice-commands): "go to the yellow sphere") but not typed (`landmarks.find` turns a
typed phrase into a marker, but nothing calls it yet).
