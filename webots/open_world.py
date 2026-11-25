"""Open world: the chair does not get the building's floor plan, or a list of its rooms. It reads the
directory on the wall, then finds its way by the signs, as a visitor would.

    python webots/run.py fp_hospital --open-world                         # it finds a directory, you pick
    python webots/run.py fp_hospital --open-world --target "Medical Imaging"   # as an appointment letter says
    python webots/open_world.py fp_hospital                               # the same as run.py ... --open-world

It runs inside run.py, so the people, the chair and its sensors (the lidar is the usual 4 m one) and
every option (--phones, --low-obstacles, --sensors, --crowd-scale, --no-stream, --driver, --model,
--view ...) are exactly those of an ordinary trip. Only the floor plan is taken away.

1. The chair starts knowing nothing (explore.py) and stands still, with the shortest path to the
   nearest directory drawn on the floor (orange), as a visitor is shown the way to it. G drives that
   path and holds still in front of the directory until it is read (vision.py); or drive there
   yourself (O). The directory lists the departments and public places with their wing. A home has
   none: G explores, reads the room signs it spots and lists the places it has read instead.
2. The rider picks one of them (1-9, then G). As a visitor notes the wing, the directory is read once:
   the chair does not go up to (or read) other directories on the way. It explores towards the place:
   signs for its wing first ("East Wing"), then for the department, going up to signs it has spotted
   but not read, keeping near rooms of the department once it has read some, until it reads the
   department's or place's own sign, and stops there.
3. There, the places it has read close by are listed (D switches between them and the directory):
   pick a room and G again (Ward 4A, then 4A Room 2).
The learned assistant (RL) drives all the way. The goal post only stands on a real destination (the
directory, or a place whose sign it has read), never on the spots it explores. A trip does not end when
the chair gets somewhere or bumps someone: R starts again (the same start and people, the chair
knowing nothing again), N opens the next building. Keys, besides run.py's:
    1-9     pick from the list on the screen (the directory, or the places read nearby)
    P       next page of the list
    D       switch the list: the directory / the places read nearby
    G       go to the one picked (nothing picked and no directory read: go to the directory)
    C       stop where you are
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

WEBOTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBOTS_DIR.parent))

from envs.floorplan import ROUTE_CELL_M  # noqa: E402

PICK_EVERY_S = 3.0          # how often to think again about which frontier to explore
ARRIVED_M = 0.8             # this close to a destination's stopping spot is there
FRONTIER_REACHED_M = 1.2
STUCK_S, STUCK_M = 12.0, 0.4
NOTHING_LEFT_S = 8.0        # no frontier it can reach for this long: it has explored everywhere
POCKET_M2 = 25.0            # the chair can reach less floor than this: it is boxed in, not done
BACK_OUT_M, BACK_OUT_S = 2.0, 20.0   # wedged in a gap: back out this far along its own trail (for at most this long)
CREEP_SPEED, CREEP_ROOM_M = 0.3, 0.55   # ... slowly, and only with this much room in front (from the lidar)
WAY_OUT_M = 1.5             # backing out: this far past the gap it came through
TRAIL_M = 0.3               # where the chair has been, a point every this many metres
MAP_EVERY_STEPS = 2         # lidar scans added to the chair's map: every this many steps (0.2 s)
LOOK_FROM_M = {"plate": (2.0, 1.5), "blade": (3.0, 2.0), "direction": (4.0, 3.0), "directory": (1.7, 2.1, 2.6)}   # look closer from here
LOOK_ASIDE_RAD = math.radians(25)   # someone standing in front of a sign: look from this far to one side
LOOK_REACHED_M = 1.0
LOOK_HOLD_S = {"directory": 4.0}     # facing the sign: hold still this long for it to be read (others 2 s)
LOOK_AGAIN_S = 30.0         # a directory given up on is worth another look after this long
LOOK_GO_S, LOOK_GIVE_UP_S = 12.0, 20.0   # give a look up if the spot is not reached in this long, or all of it takes this long
DIRECTORY_GO_S = 25.0       # ... a directory is worth going further for
DIRECTORY_RETRY_S = 6.0     # a look at a directory that failed (someone in the way): again this soon, from another spot
DIRECTORY_TRIES = 3         # ... for this many looks, then it is left for LOOK_AGAIN_S
LOOK_TRIES = {"directory": 5}   # readings of a sign to wait for while facing it (others: 2) ...
LOOK_BUSY_S = 1.0           # ... while a reading is under way or was this recently (run.py reads in the background,
LOOK_HOLD_MAX_S = 15.0      # slower than the simulation), but never longer than this
OTHERS_M = 5.0              # a room sign this near another department's rooms (and not ours) is not worth a look
DIRECTORY_LOOK_M = 14.0     # an unread directory counts this much nearer than a frontier (a room sign: explore.LOOK_BONUS_M)
FIND_DIRECTORY_S = 240.0    # looking for a directory this long without finding one (a home has none): stop looking
DIRECTORY_HOLD_MAX_S = 25.0   # driven to a directory: hold still in front of it this long at most for it to be read
BOARD_FROM_M = (2.1, 1.9, 2.4)   # ... read from this far in front (the whole 1.6 m board in view, its print big enough)
BOARD_REACHED_M = 0.3       # ... from this near its reading spot (further off, the board is seen at a slant)
FACING_RAD = {"directory": 0.04}   # turned to a sign: this near straight at it (others 0.12), for the whole board
PATH_NODE = ("DEF DIRECTORY_PATH Group {{ children [ {} ] }}")
PATH_PIECE = ("Pose {{ translation {x:.3f} {y:.3f} 0.012 rotation 0 0 1 {yaw:.4f} children [ Shape {{ appearance "
              "PBRAppearance {{ baseColor 1 0.55 0 emissiveColor 1 0.55 0 emissiveIntensity 0.7 roughness 1 metalness 0 }} "
              "geometry {geometry} castShadows FALSE isPickable FALSE }} ] }}")
NEARBY_M = 15.0             # places read within this of the chair are "nearby"
BLOCKED_S, BLOCKED_M = 6.0, 2.5   # not moving for this long with someone this near: say so
LOOK_ROUND_RAD_S = 0.6      # at the start the chair turns once on the spot, as a visitor looks round
KEY_HELP = [                # run.py's list, with the directory keys instead of its marker ones
    "O       take control / hand back",
    "Arrows  drive while you have control (no keys = stop)",
    "Space   pause / resume",
    "M       camera: around the chair / trip (as in 2D) / whole building",
    "F       first person: the rider's eye, then each phone walker's, then back",
    "S       what the sensors see on / off",
    "1-9 P G pick from the list / next page / go (before a directory is read: to it)",
    "D C     switch the list (directory or places read nearby) / stop",
    "L H     show / hide this list / the text at the top",
    "R N Q   start again (same start, knowing nothing) / next building / quit",
]
KEY_HELP_MARKERS = [        # the same with --landmarks: markers always up, the red line while the rider drives
    "O       take control / hand back",
    "Arrows  drive while you have control, aiming the red line at a marker",
    "1-8 G   while you have control: pick a marker / go to it (or the red post)",
    "1-9 P G otherwise: pick from the list / next page / go (no directory yet: to it)",
    "D C     switch the list (directory or places read nearby) / stop",
    "Space M pause / camera: around the chair, trip, whole building",
    "F       first person: the rider's eye, then each phone walker's, then back",
    "S       what the sensors see on / off",
    "L H     show / hide this list / the text at the top",
    "R N Q   start again (same start, knowing nothing) / next building / quit",
]


MARKER_SPACING_M = 6.0      # the open world's markers: one every this far along the walls, the whole building
MARKER_LIMIT = 48           # ... at most (8 colours by 6 shapes, each pair once)
MARKERS_LISTED = 9          # the nearest ones in sight are numbered (1-9), for the keys, the red line and voice
MARKER_SEEN_M = (1.0, 30.0)  # ... from this near to this far, with nothing in the way
MARKER_SIGN_GAP_M = 1.5     # never this near a sign, so a marker does not hide one from the rider
MARKER_SHAPES = {           # twice the size of the --landmarks ones (about 60 cm), to be seen down a corridor and from above
    "sphere": "Sphere { radius 0.30 subdivision 2 }",
    "cube": "Box { size 0.55 0.55 0.55 }",
    "cone": "Cone { bottomRadius 0.32 height 0.64 }",
    "cylinder": "Cylinder { radius 0.26 height 0.60 }",
    "pyramid": "Cone { bottomRadius 0.40 height 0.60 subdivision 4 }",
    "capsule": "Capsule { radius 0.19 height 0.40 }",
}


def building_markers(plan, world: str, signs_at=()) -> list:
    """Markers for the whole building, put up once and left there, like the art and features a
    visitor finds their way by: along the walls, near floor the chair can reach, one about every
    MARKER_SPACING_M (the farthest spot from those so far each time), the same for every run of a
    world. Each has its own colour and shape pair ("red pyramid"), and markers near each other are
    given different colours where they can be seen together, so "the red one" is usually enough.
    None goes within MARKER_SIGN_GAP_M of a sign in `signs_at` (x, y), so none hides one."""
    import random

    import cv2

    from envs.floorplan import FREE
    from webots import landmarks
    from webots.people import line_of_sight
    free = (plan.cells == FREE).astype(np.uint8)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * plan.resolution
    ring = (clearance > landmarks.WALL_GAP_M - plan.resolution) & \
        (clearance <= landmarks.WALL_GAP_M + plan.resolution) & (landmarks.reachable_area(plan) > 0)
    rows, cols = np.nonzero(ring)
    if len(rows) == 0:
        return []
    keep = np.arange(0, len(rows), 4)
    points = np.stack([cols[keep] * plan.resolution + plan.resolution / 2 - plan.width_m / 2,
                       plan.height_m / 2 - (rows[keep] * plan.resolution + plan.resolution / 2)], axis=1)
    for sx, sy in signs_at:
        points = points[np.hypot(points[:, 0] - sx, points[:, 1] - sy) >= MARKER_SIGN_GAP_M]
    if len(points) == 0:
        return []
    rng = random.Random(f"open world markers {world}")
    chosen = [rng.randrange(len(points))]
    gap = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < MARKER_LIMIT and gap.max() >= MARKER_SPACING_M:
        chosen.append(int(gap.argmax()))
        gap = np.minimum(gap, np.linalg.norm(points - points[chosen[-1]], axis=1))
    pairs = [(c, sh) for c in landmarks.PALETTE for sh in MARKER_SHAPES]
    rng.shuffle(pairs)
    named: list = []
    for k in chosen:
        x, y = points[k]
        near = [lm.colour for lm in named if math.hypot(lm.x - x, lm.y - y) <= MARKER_SEEN_M[1]
                and line_of_sight(plan, (x, y), (lm.x, lm.y))]                  # the markers seen with it
        colour, shape = min(pairs, key=lambda p: near.count(p[0]))      # the colour least used nearby
        pairs.remove((colour, shape))
        named.append(landmarks.Landmark(f"{colour} {shape}", colour, shape, float(x), float(y)))
    return named


def marker_node(name: str, lm, height_m: float) -> str:
    """A marker as landmarks.py draws one (a shape hung on the wall at eye height, a disc of its
    colour on the floor below), bigger, and with the open world's extra shapes."""
    r, g, b = lm.rgb
    colour = f"{r:.2f} {g:.2f} {b:.2f}"
    return f"""DEF {name} Pose {{
  translation {lm.x:.3f} {lm.y:.3f} 0
  children [
    Pose {{ translation 0 0 {height_m} children [ Shape {{
      appearance PBRAppearance {{ baseColor {colour} emissiveColor {colour} emissiveIntensity 0.7 roughness 0.6 metalness 0 }}
      geometry {MARKER_SHAPES[lm.shape]} }} ] }}
    Pose {{ translation 0 0 0.006 children [ Shape {{
      appearance PBRAppearance {{ baseColor {colour} transparency 0.35 roughness 1 metalness 0 }}
      geometry Cylinder {{ height 0.01 radius 0.45 }} }} ] }}
  ]
}}"""


class BuildingMarkers:
    """The open world's markers: all of them up for the whole trip, whoever drives, on the screen only
    (the chair's camera does not see them, so none hides a sign). `in_sight` gives the ones the rider
    can pick from where the chair is now."""

    def __init__(self, scene, world: str, signs_at=()):
        from webots import landmarks
        self.scene, self.items = scene, building_markers(scene.plan, world, signs_at)
        children = scene.sup.getRoot().getField("children")
        for i, lm in enumerate(self.items):
            name = f"OPEN_MARKER{i}"
            if scene.sup.getFromDef(name) is None:   # once a Webots: R starts the trip again with them still up
                children.importMFNodeFromString(-1, marker_node(name, lm, landmarks.HEIGHT_M))
            scene.only_on_screen(scene.sup.getFromDef(name))

    def in_sight(self, x: float, y: float, heading: float, limit: int = MARKERS_LISTED) -> list:
        """The nearest markers a chair at (x, y) can see, numbered from its left to its right."""
        from webots.people import line_of_sight
        near, far = MARKER_SEEN_M
        seen = sorted((math.hypot(lm.x - x, lm.y - y), lm) for lm in self.items
                      if near <= math.hypot(lm.x - x, lm.y - y) <= far and line_of_sight(self.scene.plan, (x, y), (lm.x, lm.y)))
        shown = [lm for _, lm in seen[:limit]]

        def bearing(lm):
            return (math.atan2(lm.y - y, lm.x - x) - heading + math.pi) % (2 * math.pi) - math.pi
        return sorted(shown, key=bearing, reverse=True)


TRAIL_PIECE = 100           # the blue line behind the chair: a new piece every this many points (5 m)
TRAIL_PIECES = 400          # ... and the oldest go once there are this many (2 km of trail)


class Trail:
    """The blue line behind the chair in the open world (bridge.py draws the same line for an ordinary
    trip). Webots redraws a line whole whenever a point is added, so one line that grows for the
    whole of an open-world trip, which has no end, made every step slower: after 3000 steps adding a
    point took 78 ms. Here it is drawn in short pieces, so a point costs the same at any length.
    Each piece is made with all its points at once (at its start, joined to nothing yet), and a new
    point only changes one of them. Webots makes that change at the next step, along with the rest,
    while adding a point made Python wait until Webots had finished drawing: 30 ms a step while the
    chair moved, on a computer without a graphics card, against under 1 ms now."""

    def __init__(self, scene):
        self.scene, self.pieces, self.points, self.last, self.made, self.drawn = scene, [], None, None, 0, 0

    def _piece(self, start) -> None:
        from webots import bridge
        name = f"OPEN_TRAIL{self.made}"
        self.made += 1
        x, y = start
        here = f"{x:.3f} {y:.3f} {bridge.TRAIL_HEIGHT_M}"
        node = bridge.TRAIL_NODE.replace("DEF TRAIL_POSE", f"DEF {name}", 1).replace("DEF TRAIL ", f"DEF {name}_LINE ", 1)
        node = node.replace("point [ 0 0 -1, 0 0 -1 ]", "point [ " + ", ".join([here] * TRAIL_PIECE) + " ]")
        node = node.replace("coordIndex [ 0 1 ]", "coordIndex [ 0 0" + " -1" * (TRAIL_PIECE - 2) + " ]")   # -1: not yet
        sup = self.scene.sup
        sup.getRoot().getField("children").importMFNodeFromString(-1, node)
        self.scene.only_on_screen(sup.getFromDef(name))
        line = sup.getFromDef(f"{name}_LINE").getField("geometry").getSFNode()
        self.points, self.index = line.getField("coord").getSFNode().getField("point"), line.getField("coordIndex")
        self.count = 1                               # point 0 is where the last piece ended
        self.pieces.append(name)
        while len(self.pieces) > TRAIL_PIECES:
            old = sup.getFromDef(self.pieces.pop(0))
            if old is not None:
                old.remove()

    def add(self, x: float, y: float) -> None:
        """The chair's position, once it has moved 5 cm since the last point."""
        from webots import bridge
        if self.last is not None and math.dist(self.last, (x, y)) < 0.05:
            return
        if self.points is None or self.count >= TRAIL_PIECE:
            self._piece(self.last or (x, y))           # the new piece starts where the last one ended
        self.points.setMFVec3f(self.count, [x, y, bridge.TRAIL_HEIGHT_M])
        self.index.setMFInt32(self.count, self.count)    # joined to the point before
        self.count += 1
        self.drawn += 1
        self.last = (x, y)

    def clear(self) -> None:
        for name in self.pieces:
            node = self.scene.sup.getFromDef(name)
            if node is not None:
                node.remove()
        self.scene._screen_only = [k for k in self.scene._screen_only if k not in self.pieces]
        self.pieces, self.points, self.last, self.drawn = [], None, None, 0


class Session:
    """The chair in an open world: its map, its sign reader and what it is doing. step() is one 0.1 s
    step of everything; run.py (--open-world, through OpenWorld) and the evaluation (open_eval.py)
    both use it. The chair's lidar is left as it is (4 m, as in every trip): the map is built from it."""

    def __init__(self, scene, driver, layout, read: str = "text", wait_for_ocr: bool = False, markers: bool = True,
                 ocr_s: float = 0.0):
        """`layout` is the building's signs, for hanging them and for --read tags (names from the tags);
        the chair itself learns places only by reading them."""
        from webots.explore import ChairMap, Explorer
        from webots.vision import SignReader
        self.scene, self.driver, self.layout = scene, driver, layout
        self.reader = SignReader(scene, layout, read=read, wait_for_ocr=wait_for_ocr, ocr_s=ocr_s)
        self.ChairMap, self.Explorer = ChairMap, Explorer
        self.markers = markers

    def reset(self, seed: int) -> None:
        """A new trip: the scene's (people, where the chair starts) and the chair knowing nothing."""
        self.scene.reset(seed=seed)
        self.driver.reset()
        self.begin()

    def begin(self, still: bool = False) -> None:
        """The chair knows nothing of the building again (run.py has just reset the scene). `still`
        (run.py): it stands still with the path to the nearest directory drawn, until G (to_directory).
        Otherwise (open_eval.py) it looks round and goes looking for a directory straight away."""
        sc = self.scene
        sc.time_limit = math.inf
        self.map = self.ChairMap(sc.plan)
        self.explorer = self.Explorer()
        self.reader.seen.clear()
        self.reader.pending.clear()
        self.mode, self.target, self.goal, self.department, self.wing = "finding a directory", None, None, None, None
        self.frontier, self.picked_at, self.nothing_since = None, -math.inf, None
        self.pointers_used, self.trail = frozenset(), [(sc.robot.x, sc.robot.y)]
        self.ticks, self.stuck_from = 0, (sc.sup.getTime(), (sc.robot.x, sc.robot.y))
        self.path_m, self.last_xy, self.collisions = 0.0, (sc.robot.x, sc.robot.y), 0
        self.arrived_at, self.started_at = None, sc.sup.getTime()
        self.marked: set[str] = set()
        self.looking_at, self.looked, self.near_used, self._dept = None, {}, frozenset(), {}
        self.no_directory, self.room_signs = False, False   # room_signs: explored everywhere, none found
        self.turned, self.heading = 0.0, sc.robot.orientation   # looking round: how far it has turned
        self.still = (sc.sup.getTime(), (sc.robot.x, sc.robot.y))   # since when it has not moved (STUCK_M)
        self.backing = None                          # (point on its trail, since when): backing out of a gap
        self.look_tries = {}                         # tag -> looks at it that failed
        self.board, self.board_spot, self.board_m, self.board_tries = None, None, 0.0, 0   # the directory driven to
        self.marker = None                           # (name, spot): the marker the rider sent it to (run.py --landmarks)
        self._scan()
        self._use_map(force=True)
        self._go(None)
        if still:
            self.mode, self.turned = "start", 2 * math.pi
            self._show_path()
        else:
            self._draw_path(None)

    # ------------------------------------------------------------ what the rider asks
    def directory(self) -> dict:
        """What the directories read so far list: name -> wing."""
        return self.reader.directory()

    def read_a_directory(self) -> bool:
        """Whether a whole directory has been read (not just a few of its entries)."""
        return self.reader.have_directory()

    def go_to(self, name: str, department: str | None = None) -> None:
        """Find `name`: a directory entry (a department, or a place of its own), or a room (picked from
        the places read nearby, or from an appointment letter); `department` if known."""
        from webots.explore import same_name
        listed = next((n for n in self.directory() if same_name(n, name)), None)
        self.target = listed or name
        self.department = listed or department or self.department_of(name) or name   # itself, if nothing says
        self.wing = None
        self.mode, self.frontier, self.nothing_since, self.looking_at = "looking", None, None, None
        self.arrived_at, self.started_at, self.picked_at = None, self.scene.sup.getTime(), -math.inf
        self.path_m, self.collisions = 0.0, 0
        self._draw_path(None)

    def go_to_marker(self, name: str, spot) -> None:
        """G at a marker the rider aimed at (run.py --landmarks): drive to the spot in front of it on the
        chair's own map, reading signs on the way, and wait there. The directory and the places read
        so far are kept."""
        self.target, self.department, self.wing, self.looking_at, self.frontier = None, None, None, None, None
        self.mode, self.backing, self.marker = "to a marker", None, (name, tuple(spot))
        now, r = self.scene.sup.getTime(), self.scene.robot
        self.arrived_at, self.started_at, self.stuck_from = None, now, (now, (r.x, r.y))
        self._draw_path(None)
        self._go(self.marker[1], show=True)

    def find_directory(self) -> None:
        self.target, self.department, self.wing, self.looking_at = None, None, None, None
        self.mode, self.frontier, self.picked_at, self.nothing_since = "finding a directory", None, -math.inf, None
        self.started_at = self.scene.sup.getTime()
        self._draw_path(None)

    def to_directory(self) -> None:
        """G before a directory is read: drive the path drawn to the nearest directory (planned on the
        building's floor plan, as a visitor is shown the way) and hold still in front of it until it
        is read. A building with none (a home): explore and read the room signs instead."""
        if not self._show_path():
            self.find_directory()
            self.room_signs, self.turned = True, 0.0     # look round first, then explore
            return
        self.target, self.department, self.wing, self.looking_at, self.frontier = None, None, None, None, None
        self.mode, self.board_tries, self.backing = "to the directory", 0, None
        self.started_at = self.scene.sup.getTime()
        self._go(self.board_spot, show=True)

    def _show_path(self) -> bool:
        """Draw the shortest path from the chair to the nearest directory's reading spot (the
        building's floor plan, as a 'You are here' map shows it). False if there is no directory."""
        from envs.floorplan import route_length
        sc = self.scene
        here = (sc.robot.x, sc.robot.y)
        best = None
        for face in self.layout.faces:
            if face.kind != "directory":
                continue
            spot = self._board_spot(face, 0.0)
            route = sc.plan.route(here, spot)
            if route is not None and (best is None or route_length(here, route) < best[0]):
                best = (route_length(here, route), face, spot, [here] + list(route))
        if best is None:
            self.board = None
            self._draw_path(None)
            return False
        self.board_m, self.board, self.board_spot, points = best
        self._draw_path(points)
        return True

    def _board_spot(self, face, off: float):
        """Where to read directory `face` from: in front of it (`off` radians to one side), as near as
        BOARD_FROM_M allows with room for the chair, so that the whole board is in the camera's view."""
        plan = self.scene.plan
        for d in BOARD_FROM_M:
            spot = (face.x + math.cos(face.yaw + off) * d, face.y + math.sin(face.yaw + off) * d)
            if plan.clearance_at(*spot) >= 0.5:
                return spot
        return tuple(face.stop)

    def _draw_path(self, points) -> None:
        """An orange line on the floor along `points`, with a disc at its end. None takes it away."""
        if not self.markers:
            return
        sup = self.scene.sup
        old = sup.getFromDef("DIRECTORY_PATH")
        if old is not None:
            old.remove()
        if not points:
            return
        pieces = []
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            if length > 0.01:
                pieces.append(PATH_PIECE.format(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2,
                                                yaw=math.atan2(b[1] - a[1], b[0] - a[0]),
                                                geometry=f"Box {{ size {length + 0.1:.3f} 0.1 0.004 }}"))
        x, y = points[-1]
        pieces.append(PATH_PIECE.format(x=x, y=y, yaw=0.0, geometry="Cylinder { height 0.006 radius 0.3 }"))
        sup.getRoot().getField("children").importMFNodeFromString(-1, PATH_NODE.format(" ".join(pieces)))
        self.scene.only_on_screen(sup.getFromDef("DIRECTORY_PATH"))

    def stop(self) -> None:
        self.target, self.mode, self.frontier, self.looking_at = None, "waiting", None, None
        self._go(None)

    def department_of(self, name: str) -> str | None:
        """The directory entry `name` belongs to, as a visitor would tell: itself, or the department
        its name says ("4A Room 3": Ward 4A, "CT Scan": Medical Imaging). None if no directory read says."""
        from webots.explore import departments_for, same_name
        if name in self._dept:
            return self._dept[name]
        wanted = departments_for(name)
        found = next((n for n in self.directory() if any(same_name(n, w) for w in wanted)), None)
        if found is not None:
            self._dept[name] = found
        return found

    def nearby(self) -> list[str]:
        """Places read close to the chair, nearest first (not directory entries far off). With no
        directory, every place read is all the rider has to go by."""
        here = (self.scene.robot.x, self.scene.robot.y)
        reach = math.inf if self.no_directory and not self.directory() else NEARBY_M
        places = [(math.dist(here, stop), name) for name, stop in self.reader.places().items()
                  if math.dist(here, stop) <= reach]
        return [name for _, name in sorted(places)]

    # ------------------------------------------------------------ one step
    def step(self, command=None):
        """Drive one step: the assistant's command, or `command` (the rider's) if given. Returns what
        the scene's step returns (observation, reward, terminated, truncated, info)."""
        sc = self.scene
        auto = self.driver.act(None)                 # always, so the assistant keeps tracking people
        looking_round = self.turned < 2 * math.pi
        if command is None and looking_round:
            command = np.array([0.0, 0.0, LOOK_ROUND_RAD_S], dtype=np.float32)
        if command is None and self.backing is not None and self.backing[0]:
            command = self._creep(self.backing[0][0])
        if command is None and self.looking_at is not None and self.looking_at["phase"] == "face":
            command = self._face()
        if command is None:
            command = auto if self.goal is not None else np.zeros(3, dtype=np.float32)
        out = sc.step(command)
        info = out[4]
        self.ticks += 1
        r = sc.robot
        self.path_m += math.dist(self.last_xy, (r.x, r.y))
        self.last_xy = (r.x, r.y)
        if math.dist(self.still[1], self.last_xy) > STUCK_M or looking_round or \
                self.mode in ("start", "waiting", "arrived", "done", "at a marker"):
            self.still = (sc.sup.getTime(), self.last_xy)
        if looking_round:
            d = r.orientation - self.heading
            self.turned += abs(math.atan2(math.sin(d), math.cos(d)))
        self.heading = r.orientation
        if not self.trail or math.dist(self.trail[-1], self.last_xy) >= TRAIL_M:
            self.trail.append(self.last_xy)
        self.collisions += bool(info["COLLISION"])
        if self.ticks % MAP_EVERY_STEPS == 0:
            self._scan()
        new = self.reader.look()
        if new and self.markers:
            self._mark()
        self._use_map()
        self._decide()
        return out

    def _scan(self) -> None:
        r = self.scene.robot
        self.map.update(r.x, r.y, r.orientation, self.scene.lidar.getRangeImage(), self.scene.lidar.getMaxRange())

    def _use_map(self, force: bool = False) -> None:
        """Give the driver the chair's own map instead of the floor plan, and plan again if the
        route now runs into something."""
        sc = self.scene
        own = self.map.plan((sc.robot.x, sc.robot.y), sc.sup.getTime())
        self.reader.map_plan = own
        plan = sc.plan if self.mode == "to the directory" else own    # the path it was shown
        base = self._base()
        if base.plan is plan and not force:
            return
        base.plan = plan
        path = base.path
        if path and any(plan.clearance_at(*p) < 0.3 for p in _along(path[base.progress:], 0.25)):
            base.path = None

    def _base(self):
        return getattr(getattr(self.driver, "controller", self.driver), "base", self.driver)

    def _go(self, xy, show: bool = False) -> None:
        """Drive to `xy` (None: stop). The goal post stands there only if `show`: a real destination
        (the directory, a place read), not a spot it explores or looks at a sign from."""
        sc = self.scene
        self.goal = xy
        r = sc.robot
        gx, gy = xy if xy is not None else (r.x, r.y)
        r.goal_x, r.goal_y = gx, gy
        sc.goal_node.getField("translation").setSFVec3f([gx, gy, 0.01 if xy is not None and show else -5.0])
        self._base().path = None
        if xy is not None:
            sc.route = [(r.x, r.y), (gx, gy)]
            if sc.stream_people:
                sc.crowd.stream(force=True)

    def found(self, name: str):
        """Where to stop for `name`, if its sign has been read. A department is there once its own sign
        is read, or a room whose name says it is in it (the nearest): (name read, stop) or None."""
        from webots.explore import departments_for, norm, same_name
        places = self.reader.places()
        for read, stop in places.items():
            if same_name(read, name):
                return read, stop
        if self.department and same_name(self.department, name):
            here = (self.scene.robot.x, self.scene.robot.y)
            inside = [(math.dist(here, stop), read, stop) for read, stop in places.items()
                      if norm(name) in departments_for(read)]
            if inside:
                _, read, stop = min(inside)
                return read, stop
        return None

    def _decide(self) -> None:
        sc = self.scene
        now = sc.sup.getTime()
        here = (sc.robot.x, sc.robot.y)
        if self.mode in ("start", "to the directory") and self.read_a_directory():   # driven there, or by the rider
            self.mode, self.looking_at, self.backing = "waiting", None, None
            self._go(None)
            self._draw_path(None)
            return
        if self.mode in ("start", "waiting", "arrived", "done", "at a marker"):
            return
        if self.turned < 2 * math.pi:                # still looking round
            self.stuck_from, self.started_at = (now, here), now
            return
        if self.backing is not None:                 # backing out of a gap: until out, or given up
            back, since = self.backing
            while back and math.dist(here, back[0]) < 0.3:
                back.pop(0)
            if back and now - since < BACK_OUT_S:
                return
            self.backing, self.picked_at, self.stuck_from = None, -math.inf, (now, here)
            self._go(None)
        if self.mode == "to the directory":
            self._read_board(now, here)
            return
        if self.mode == "to a marker":
            if self.goal is None:                    # after backing out of a gap: on again
                self._go(self.marker[1], show=True)
            if math.dist(here, self.marker[1]) < ARRIVED_M:
                self.mode, self.arrived_at = "at a marker", now
                self._go(None)
                return
            self._unstick(now, here, explore=False)
            return
        if self.mode == "finding a directory":
            if self.read_a_directory():              # read one: wait for the rider to pick
                self.mode, self.looking_at = "waiting", None
                self._go(None)
                return
            if now - self.started_at > FIND_DIRECTORY_S:
                self.mode, self.no_directory, self.looking_at = "waiting", True, None
                self._go(None)
                return
        if self.target is not None and self.wing is None and self.department:
            self.wing = self.directory().get(self.department) or None   # a directory read on the way says
        if self.target is not None:
            hit = self.found(self.target)
            if hit is not None:
                name, stop = hit
                self.looking_at = None
                if self.mode != "going" or self.goal is None or math.dist(self.goal, stop) > 0.3:
                    self.mode = "going"
                    self._go(stop, show=True)
                if math.dist(here, stop) < ARRIVED_M:
                    self.mode, self.arrived_at = "arrived", now
                    self._go(None)
                    return
                self._unstick(now, here, explore=False)
                return
        # explore: keep the frontier being driven to until it is reached or mapped, or a direction sign
        # for the destination is read (then think again: it may point elsewhere)
        if self.looking_at is not None and self._look_closer(now, here):
            return
        plan = self._base().plan
        pointers, near = self._pointers(), self._near()
        news = pointers != self.pointers_used or frozenset(near) != self.near_used
        if self.frontier is not None and math.dist(here, self.frontier) < FRONTIER_REACHED_M:
            self.frontier, self.picked_at = None, -math.inf    # reached: pick the next one now
        if news or now - self.picked_at > (1.0 if self.frontier is None else PICK_EVERY_S):
            frontiers = self.map.frontiers(plan)
            self.explorer.now = now
            current = self.frontier
            looks = self._unread(plan)
            if current is not None and not news and any(math.dist(current, (x, y)) < 1.0 for x, y, _ in frontiers):
                pick = current
            else:
                finding = self.mode == "finding a directory"      # directories hang at the main reception
                pick = self.explorer.choose(plan, frontiers, here, self.reader.hints(),
                                            "Main Reception" if finding else self.target, current=current,
                                            department=self.department, looks=[(x, y, bonus) for (x, y), _, bonus in looks],
                                            near=list(near.values()), wing=self.wing)
            self.pointers_used, self.near_used, self.picked_at = pointers, frozenset(near), now
            look = next((tag for xy, tag, _ in looks if xy == pick), None)
            if look is not None:                     # a sign spotted but not read: go and look at it
                s = self.reader.seen[look]
                self.looking_at = dict(tag=look, spot=pick, sign=(s.x, s.y), phase="go", since=now, facing=None)
                self.frontier = pick
                self._go(pick)
                return
            boxed_in = float((~plan.blocked).sum()) * ROUTE_CELL_M ** 2 < POCKET_M2
            if pick is None and boxed_in:
                # Nothing reachable from a small patch: the chair has driven in between furniture, or
                # someone stands in the doorway, and its map has closed the gap behind it. It goes
                # back out the way it came (the driver heads straight there when there is no route).
                pick = self._way_out(plan, here)
            if pick is None and any(self.explorer.resting((x, y)) for x, y, _ in frontiers):
                self.nothing_since = None            # only given up on for now: try those again later
            elif pick is None:
                # nothing left to explore: done, unless that is only for now (someone standing in a
                # doorway is a wall in the chair's map until they move)
                self.nothing_since = now if self.nothing_since is None else self.nothing_since
                if now - self.nothing_since >= NOTHING_LEFT_S:
                    if self.mode == "finding a directory" and not self.room_signs:
                        self.room_signs, self.nothing_since, self.picked_at = True, None, -math.inf
                        return                       # none anywhere (a home): read the room signs instead
                    if self.mode == "finding a directory":
                        self.mode, self.no_directory, self.frontier = "waiting", True, None
                    else:
                        self.mode, self.frontier = "done", None
                    self._go(None)
                    return
            else:
                self.nothing_since = None
            if pick is not None and pick != self.frontier:   # a new one, or the old one was mapped meanwhile
                self.frontier = pick
                self._go(pick)
        self._unstick(now, here, explore=True)

    def _read_board(self, now, here) -> None:
        """Driving to the directory: at its reading spot, turn to it and hold still until it is read
        (the reader then says so); not read after a while (someone in the way): from one side, then
        the other; then stop and say so."""
        b, L = self.board, self.looking_at
        if L is None:
            if self.goal is None:                    # after backing out of a gap: on again
                self._go(self.board_spot, show=True)
            if math.dist(here, self.board_spot) < BOARD_REACHED_M:
                self.looking_at = dict(tag=b.tag, kind="directory", spot=self.board_spot, sign=(b.x, b.y),
                                       phase="face", since=now, facing=None)
                self._go(None)
                return
            self._unstick(now, here, explore=False)
            return
        if L["facing"] is None and now - L["since"] > 10.0:     # cannot turn quite straight at it: as it is
            L["facing"], L["tries0"] = now, 0
        if L["facing"] is None or now - L["facing"] < DIRECTORY_HOLD_MAX_S:
            return
        self.board_tries += 1
        self.looking_at = None
        if self.board_tries >= DIRECTORY_TRIES:
            self.mode = "waiting"
            self._go(None)
            return
        self.board_spot = self._board_spot(b, LOOK_ASIDE_RAD if self.board_tries == 1 else -LOOK_ASIDE_RAD)
        self._go(self.board_spot, show=True)

    def _wedged(self, here) -> bool:
        """The chair stands where its own map, now that it has seen the place, says it cannot be."""
        plan = self._base().plan
        r, c = plan.cell_of(*here)
        rows, cols = plan.blocked.shape
        return 0 <= r < rows and 0 <= c < cols and bool(plan.blocked[r, c])

    def _back(self, here) -> list:
        """Its own trail back the way it came, BACK_OUT_M of it (it drove there, so it fits)."""
        out = []
        for p in reversed(self.trail):
            if math.dist(p, here) < 0.3 and not out:
                continue
            out.append(p)
            if math.dist(p, here) >= BACK_OUT_M:
                break
        return out

    def _creep(self, to) -> np.ndarray:
        """Back out along its trail without the route follower, which brakes for walls the chair's
        map puts around it in a tight gap: turn on the spot towards the next trail point, then edge
        forward while the lidar shows room in front (people included)."""
        r, sc = self.scene.robot, self.scene
        a = math.atan2(to[1] - r.y, to[0] - r.x) - r.orientation
        a = math.atan2(math.sin(a), math.cos(a))
        turn = float(np.clip(2.0 * a, -0.6, 0.6))
        if abs(a) > 0.35:
            return np.array([0.0, 0.0, turn], dtype=np.float32)
        ranges = np.asarray(sc.lidar.getRangeImage(), dtype=np.float32)
        n = len(ranges)
        cone = [n // 2 + k for k in range(-n // 12, n // 12 + 1)]      # 30 degrees either side of ahead
        room = float(np.nanmin(np.where(np.isfinite(ranges[cone]), ranges[cone], np.inf)))
        forward = CREEP_SPEED if room > CREEP_ROOM_M else 0.0
        return np.array([forward, 0.0, turn], dtype=np.float32)

    def _way_out(self, plan, here):
        """A spot on the chair's own trail just outside the patch it is boxed into, or None."""
        open_ = ~plan.blocked
        rows, cols = open_.shape

        def inside(p):
            r, c = plan.cell_of(*p)
            return bool(open_[max(r - 2, 0):min(r + 3, rows), max(c - 2, 0):min(c + 3, cols)].any())

        trail = self.trail
        for i in range(len(trail) - 1, -1, -1):
            if math.dist(trail[i], here) > 1.0 and not inside(trail[i]):
                j = i                                # and a little further back, clear of the gap
                while j > 0 and math.dist(trail[j], trail[i]) < WAY_OUT_M:
                    j -= 1
                return trail[j]
        return None

    # ------------------------------------------------------------ looking closer at a sign
    def _unread(self, plan) -> list:
        """[((x, y), tag, bonus)]: where to look from at each sign spotted (its tag, twice) but not read
        yet, and how much nearer than it is it counts. Looking for a directory, only directories and
        direction signs are worth it. Looking for a place, room signs among another department's rooms
        are left: rooms of a department are together, so the place is not likely to be one of those."""
        from webots.explore import LOOK_BONUS_M
        now = self.scene.sup.getTime()
        seen = self.reader.seen.values()
        read = [(s.x, s.y) for s in seen if s.read and s.kind in ("plate", "blade")]
        ours, others = self._department_rooms()
        finding = self.mode == "finding a directory" and not self.room_signs
        noted = self.read_a_directory()     # as a visitor: the directory is read once, then the wings' signs lead
        rows, cols = plan.blocked.shape
        out = []
        for s in seen:
            if s.read or s.times < 2 or (noted and s.kind == "directory"):
                continue
            if s.tag in self.looked and (s.kind != "directory" or now - self.looked[s.tag] < LOOK_AGAIN_S):
                continue
            if finding and s.kind in ("plate", "blade"):
                continue
            if s.kind in ("plate", "blade") and any(math.dist((s.x, s.y), p) < 1.0 for p in read):
                continue                             # the other side of a blade sign, or its door plate
            if s.kind in ("plate", "blade") and any(math.dist((s.x, s.y), p) < OTHERS_M for p in others) and \
                    not any(math.dist((s.x, s.y), p) < OTHERS_M for p in ours):
                continue
            bonus = DIRECTORY_LOOK_M if s.kind == "directory" else LOOK_BONUS_M
            offs = (0.0, LOOK_ASIDE_RAD, -LOOK_ASIDE_RAD)
            tries = self.look_tries.get(s.tag, 0) % len(offs)
            for off in offs[tries:] + offs[:tries]:   # straight on, else (or after a failed look) from one side
                for d in (LOOK_FROM_M[s.kind] if off == 0.0 else sorted(LOOK_FROM_M[s.kind], reverse=True)):
                    spot = (s.x + math.cos(s.yaw + off) * d, s.y + math.sin(s.yaw + off) * d)
                    r, c = plan.cell_of(*spot)
                    if 0 <= r < rows and 0 <= c < cols and not plan.blocked[r, c]:
                        out.append((spot, s.tag, bonus))
                        break
                else:
                    continue
                break
        return out

    def _look_closer(self, now, here) -> bool:
        """Drive to the spot in front of the sign, turn to face it and hold still while it is read.
        False once done (read, or given up), so the chair goes on exploring."""
        L = self.looking_at
        s = self.reader.seen.get(L["tag"])
        if L["tag"] in self.reader.pending:          # a reading of it is under way (run.py reads in the background)
            L["busy"] = now
        held = False
        if L["facing"] is not None and s is not None:
            facing_s, tries = now - L["facing"], s.attempts - L["tries0"]
            reading = L["tag"] in self.reader.pending or now - L.get("busy", -math.inf) < LOOK_BUSY_S
            held = facing_s >= LOOK_HOLD_MAX_S or (facing_s >= LOOK_HOLD_S.get(s.kind, 2.0) and
                                                  (tries >= LOOK_TRIES.get(s.kind, 2) or not reading))
        directory = s is not None and s.kind == "directory"
        slow = L["facing"] is None and now - L["since"] > (     # once facing it, LOOK_HOLD_MAX_S is the limit
            (DIRECTORY_GO_S if directory else LOOK_GO_S) if L["phase"] == "go"
            else LOOK_GIVE_UP_S + (DIRECTORY_GO_S - LOOK_GO_S) * directory)
        known = s is not None and s.kind in ("plate", "blade") and any(      # read meanwhile from beside it
            o.read and o.kind in ("plate", "blade") and o.tag != s.tag and math.dist((s.x, s.y), (o.x, o.y)) < 1.0
            for o in self.reader.seen.values())
        if s is None or s.read or held or slow or known:
            self.looked[L["tag"]] = now
            if directory and not s.read:             # someone in the way: soon again, from the next spot round
                self.look_tries[L["tag"]] = self.look_tries.get(L["tag"], 0) + 1
                if self.look_tries[L["tag"]] % DIRECTORY_TRIES:   # ... but after a few, leave it a while
                    self.looked[L["tag"]] = now - LOOK_AGAIN_S + DIRECTORY_RETRY_S
            self.looking_at, self.frontier, self.picked_at = None, None, -math.inf
            return False
        moved = (s.x - L["sign"][0], s.y - L["sign"][1])     # where it is gets clearer on the way
        L["sign"] = (s.x, s.y)
        if L["phase"] == "go":
            L["spot"] = (L["spot"][0] + moved[0], L["spot"][1] + moved[1])
            if self.goal is not None and math.dist(self.goal, L["spot"]) > 0.3:
                self._go(L["spot"])
        if L["phase"] == "go" and math.dist(here, L["spot"]) < LOOK_REACHED_M:
            L["phase"] = "face"
            self._go(None)
        return True

    def _face(self) -> np.ndarray:
        """Turn on the spot towards the sign being looked at; note when it is straight ahead."""
        L, r = self.looking_at, self.scene.robot
        a = math.atan2(L["sign"][1] - r.y, L["sign"][0] - r.x) - r.orientation
        a = math.atan2(math.sin(a), math.cos(a))
        s = self.reader.seen.get(L["tag"])
        if abs(a) < FACING_RAD.get(L.get("kind") or (s.kind if s is not None else ""), 0.12):
            if L["facing"] is None:
                L["facing"], L["tries0"] = self.scene.sup.getTime(), s.attempts if s is not None else 0
            return np.zeros(3, dtype=np.float32)
        turn = math.copysign(max(abs(2.0 * a), 0.12), a)    # not so slow that the last few degrees take for ever
        return np.array([0.0, 0.0, float(np.clip(turn, -0.6, 0.6))], dtype=np.float32)

    def _near(self) -> dict:
        """Places read in the destination's department (as their names say): name -> stop."""
        if not self.department:
            return {}
        return {name: stop for name, (stop, dep) in self._read_places().items()
                if dep == self.department and name != self.target}

    def _department_rooms(self):
        """Stops of the places read in the destination's department, and in other departments."""
        if not self.department:
            return [], []
        places = self._read_places().values()
        return [stop for stop, dep in places if dep == self.department], \
               [stop for stop, dep in places if dep and dep != self.department]

    def _read_places(self) -> dict:
        """Places read: name -> (stop, the directory entry it belongs to, as its name says, or None)."""
        return {name: (stop, self.department_of(name)) for name, stop in self.reader.places().items()}

    def _pointers(self) -> frozenset:
        """The direction sign rows read so far that point the way to the destination."""
        if self.target is None:
            return frozenset()
        from webots.explore import departments_for, same_name
        wanted = departments_for(self.target, self.department) | ({self.wing} if self.wing else set())
        return frozenset((s.tag, name, arrow) for s in self.reader.seen.values() for name, arrow in s.arrows
                         if any(same_name(name, w) for w in wanted))

    def _unstick(self, now, here, explore: bool) -> None:
        t0, xy0 = self.stuck_from
        if math.dist(here, xy0) > STUCK_M:
            self.stuck_from = (now, here)
        elif now - t0 > STUCK_S:
            self.stuck_from = (now, here)
            back = self._back(here) if self._wedged(here) else None
            if back:                                 # squeezed into a gap its map now shows is too narrow:
                if explore and self.frontier is not None:    # back out the way it came, then go round
                    self.explorer.give_up(self.frontier, now)
                self.frontier, self.looking_at, self.backing = None, None, (back, now)
                self._go(None)
                return
            if explore and self.frontier is not None:
                self.explorer.give_up(self.frontier, now)
                self.frontier = None
            else:
                self._base().path = None

    def _mark(self) -> None:
        """A green ball over every sign the chair has read, where its camera put it."""
        sup = self.scene.sup
        if sup.getFromDef("READ_SIGNS") is None:
            sup.getRoot().getField("children").importMFNodeFromString(-1, "DEF READ_SIGNS Group { children [] }")
            self.scene.only_on_screen(sup.getFromDef("READ_SIGNS"))
        group = sup.getFromDef("READ_SIGNS").getField("children")
        for s in self.reader.seen.values():
            key = f"{s.tag}"
            if not s.read or key in self.marked:
                continue
            self.marked.add(key)
            z = float(np.mean(s.zs)) + 0.32
            group.importMFNodeFromString(-1, (
                f"Pose {{ translation {s.x:.3f} {s.y:.3f} {z:.3f} children [ Shape {{ appearance PBRAppearance {{ "
                f"baseColor 0.1 0.85 0.25 emissiveColor 0.1 0.85 0.25 emissiveIntensity 1 roughness 1 metalness 0 }} "
                f"geometry Sphere {{ radius 0.07 subdivision 2 }} }} ] }}"))

    # ------------------------------------------------------------ for the screen
    def status(self) -> str:
        sc = self.scene
        here = (sc.robot.x, sc.robot.y)
        if self.mode == "start":
            if self.board is None:
                return "Standing still. No directory in this building: G explores and reads the room signs"
            return (f"Standing still. The orange path is the shortest way to the directory ({self.board_m:.0f} m): "
                    f"G goes and reads it, or drive there yourself (O)")
        if self.turned < 2 * math.pi and self.mode not in ("waiting", "arrived", "done"):
            return "Looking round"
        if self.backing is not None:
            return "Backing out of a gap too narrow for it"
        facing = self.looking_at is not None and self.looking_at["phase"] == "face"   # holding still on purpose
        if self.mode not in ("waiting", "arrived", "done", "at a marker") and not facing and \
                sc.sup.getTime() - self.still[0] > BLOCKED_S and \
                any(math.dist(here, (p.x, p.y)) < BLOCKED_M for p in sc.crowd.everyone):
            return "Waiting for people to make way (O: drive past them yourself)"
        if self.mode == "to a marker":
            return f"Going to the {self.marker[0]}: {math.dist(here, self.marker[1]):.1f} m to go"
        if self.mode == "at a marker":
            then = "pick where to go (1-9, then G)" if self.read_a_directory() or self.no_directory else \
                "G goes to the directory" if self.board is not None else "G explores"
            return f"At the {self.marker[0]}: {then}, or O to aim at the next marker"
        if self.mode == "to the directory":
            if self.looking_at is not None:
                return "Reading the directory: holding still in front of it"
            return f"Going to the directory along the orange path: {math.dist(here, self.board_spot):.1f} m to go"
        if self.mode == "finding a directory":
            if self.room_signs:
                return "No directory anywhere: reading the room signs" + (" up close" if self.looking_at is not None else "")
            return "Looking for a directory" + (": reading one up close" if self.looking_at is not None else "")
        if self.mode == "waiting":
            if self.read_a_directory():
                return "Read a directory: pick where to go (1-9, then G)"
            if self.no_directory:
                return "No directory found: pick a place it has read (1-9, then G)"
            if self.board is not None:
                return ("Could not read the directory (someone in the way?): " if self.board_tries >= DIRECTORY_TRIES
                        else "Stopped: ") + "G goes to the directory"
            return "Waiting: pick where to go (1-9, then G)"
        if self.mode == "arrived":
            from webots.explore import same_name
            department = self.department and same_name(self.department, self.target) and \
                any(n != self.target for n in self.nearby())
            return f"Arrived at {self.target}" + (": pick a place read here (D, 1-9, then G)" if department else "")
        if self.mode == "done":
            return "Explored everywhere it can reach" + (f": no sign for {self.target}" if self.target else "")
        if self.mode == "going":
            return f"Going to {self.target}: read its sign, {math.dist(here, self.goal):.1f} m to go"
        where = f"Looking for {self.target}" + (f" ({self.wing})" if self.wing else "")
        if self.looking_at is not None:
            return where + ": reading a sign up close"
        from webots.explore import departments_for, same_name
        wanted = departments_for(self.target, self.department) | ({self.wing} if self.wing else set())
        pointing = [n for n, *_ in self.reader.hints() if any(same_name(n, w) for w in wanted)]
        return where + (f": following signs for {pointing[-1]}" if pointing else ": exploring")

    def knowledge(self) -> str:
        places = self.reader.places()
        directions = sum(1 for s in self.reader.seen.values() if s.kind == "direction" and s.arrows)
        directories = sum(1 for s in self.reader.seen.values() if s.kind == "directory" and s.listing)
        seen_m2 = float(self.map.known_open().sum()) * self.map.res ** 2
        return (f"Read {len(places)} places, {directions} direction signs, {directories} directories   "
                f"Seen {seen_m2:.0f} m2 of floor")


def _along(points, step):
    """Points every `step` metres along a polyline."""
    out = []
    for a, b in zip(points, points[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)]
    return out


class OpenWorld:
    """What run.py --open-world adds to an ordinary trip: the keys for the list on the screen (1-9, P, D,
    G, C), the status lines and the list itself. The list is only what the chair has read: the
    directory (each entry with its wing), or the places read nearby. With --landmarks, while the rider
    drives the number keys and G are the markers' (run.py), and G sends the chair to one (go_to_marker).
    run.py does everything else."""

    def __init__(self, scene, driver, world_stem: str, read: str = "text", target: str | None = None,
                 trail: bool = True):
        from webots import signs
        self.scene, self.driver, self.world = scene, driver, world_stem
        self.trail = Trail(scene) if trail else None      # run.py turns the scene's own trail off
        self.layout = signs.for_world(scene.sup, scene.plan, world_stem)
        self.session = Session(scene, driver, self.layout, read=read)
        self.target = target
        self.showing, self.page, self.picked, self.last_status = "directory", 0, None, ""
        print(f"{world_stem}: {len(self.layout.faces)} sign faces on the walls; the chair knows none of them yet",
              flush=True)

    def begin(self) -> None:
        """A new trip (run.py has reset the scene): the chair knows nothing again."""
        self.session.begin(still=not self.target)
        if self.trail is not None:
            self.trail.clear()
        self.showing, self.page, self.picked, self.last_status = "directory", 0, None, ""
        if self.target:                             # an appointment letter: no need to wait for a directory
            self.session.go_to(self.target)
            self.picked = self.target

    def items(self, showing: str | None = None) -> list[str]:
        """The list on the screen: the directory read (Emergency first, then A to Z), or the places read
        nearby (nearest first)."""
        if (showing or self.showing) == "directory":
            return sorted(self.session.directory(), key=lambda n: (n != "Emergency", n.lower()))
        return self.session.nearby()

    def places(self) -> list[str]:
        """Every place the chair can be sent to now: the directory read and the places read nearby."""
        return list(dict.fromkeys(self.items("directory") + self.items("nearby")))

    def pick(self, name: str) -> bool:
        """Picks a place by its name, as 1-9 does from the list (voice.py): the list turns to it.
        False if the chair has not read it."""
        for showing in dict.fromkeys((self.showing, "directory", "nearby")):
            items = self.items(showing)
            if name in items:
                self.showing, self.page, self.picked = showing, items.index(name) // 9, name
                print(f"Picked: {name} (G goes there)", flush=True)
                return True
        return False

    def names(self) -> list[str]:
        return self.items()[self.page * 9:(self.page + 1) * 9]

    def keys(self, pressed: set) -> bool:
        """The list keys. True when the chair should drive itself again (G)."""
        session = self.session
        if ord("D") in pressed:
            self.showing = "nearby" if self.showing == "directory" else "directory"
            self.page = 0
        names = self.names()
        for digit in range(1, 10):
            if ord(str(digit)) in pressed and digit <= len(names):
                self.picked = names[digit - 1]
                print(f"Picked: {self.picked} (G goes there)", flush=True)
        if ord("P") in pressed and self.items():
            self.page = (self.page + 1) % math.ceil(len(self.items()) / 9)
        if ord("G") in pressed:
            if self.picked:
                inside = session.department if session.mode == "arrived" else None   # a room of the department it is in
                session.go_to(self.picked, department=inside)
                self.showing, self.page = ("directory" if session.directory() else self.showing), 0
                return True
            if not session.read_a_directory() and not session.no_directory and \
                    session.mode in ("start", "waiting", "at a marker"):
                session.to_directory()
                return True
            print("Pick where to go first: 1 to 9 from the list on the screen", flush=True)
        if ord("C") in pressed:
            session.stop()
        return False

    def go_to_marker(self, name: str, spot) -> None:
        """G while the rider aims at a marker (run.py --landmarks): the chair drives itself there."""
        self.session.go_to_marker(name, spot)

    def step(self, command=None):
        before = self.session.mode
        out = self.session.step(command)
        if self.trail is not None:
            self.trail.add(self.scene.robot.x, self.scene.robot.y)
        if self.session.mode == "arrived" and before != "arrived" and \
                any(n != self.session.target for n in self.session.nearby()):
            self.showing, self.page, self.picked = "nearby", 0, None    # at a department: its rooms read
        if self.session.mode == "waiting" and before != "waiting" and self.session.no_directory and \
                not self.session.directory():
            self.showing, self.page, self.picked = "nearby", 0, None    # no directory: the places read
        status = self.session.status()
        steady = re.sub(r"[\d.]+ m to go", "", status)       # not a new line for every metre driven
        if steady != self.last_status:
            print(status, flush=True)
            self.last_status = steady
        return out

    def badge(self) -> str:
        mode = {"start": "standing still"}.get(self.session.mode, self.session.mode)
        return f"OPEN WORLD: {mode.upper()}  (1-9 pick, G go, D list, C stop)"

    def lines(self) -> list[str]:
        """The open world's own status lines: what it is doing, and what it knows."""
        return [self.session.status(), self.session.knowledge()]

    def legend(self, overlay) -> None:
        """The list under the status lines, the picked one in [brackets]: the directory read (each
        entry with its wing), or the places read nearby."""
        session, names = self.session, self.names()
        directory = session.directory()
        if self.showing == "directory":
            title = "Directory"
            marked = [n + (f" ({directory[n]})" if directory.get(n) else "") for n in names]
            empty = "" if directory else "nothing read yet: G goes to the directory" if session.mode in ("start", "at a marker") \
                else "nothing read yet" if session.mode == "to a marker" else "nothing read yet: the chair is looking for a directory"
        elif session.no_directory and not directory:
            title = "No directory: places read"
            marked = names
            empty = "no places read yet"
        else:
            title = "Read nearby"
            marked = names
            empty = "no places read within 15 m"
        pages = max(1, math.ceil(len(self.items()) / 9))
        overlay.legend(marked, names.index(self.picked) if self.picked in names else None,
                       (empty + "   " if not names else "") +
                       f"Page {self.page + 1}/{pages} (P)   D: {'places read nearby' if self.showing == 'directory' else 'the directory'}"
                       f"   picked: {self.picked or '-'}", title=title)


def main(argv=None) -> None:
    """open_world.py <world> [options] is run.py <world> --open-world [options], with run.py's options."""
    from webots import run
    run.main(list(sys.argv[1:] if argv is None else argv) + ["--open-world"])


if __name__ == "__main__":
    main()
