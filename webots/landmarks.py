"""Landmarks: coloured markers along the walls that a rider can refer to ("the yellow sphere").

    python webots/run.py corridor --landmarks        # 6 landmarks; --landmarks 8 for more

With `--landmarks` they are redrawn every time the rider takes the controls (O): a few new
ones, on walls a chair at that spot could see from 3 to 15 m away, so wherever the rider has got
to there are places to head for, as in a building they do not know. `Landmarks.refresh` does that.

A landmark is a small coloured shape hung on the wall at about head height, with a disc of the
same colour on the floor below it, so it shows in the overhead views and in the first-person
view. Each has a unique name made of its colour and its shape ("yellow sphere"). They are chosen
from the building's floor plan, spread out along its walls and kept near places a person can
reach, the same way for every run of a world.

They are only for looking at and for naming a place. They have no physics and no bounding
object, so the lidar does not see them and they change nothing about how the chair or the
people move. They are added to the world while we run, not saved in the .wbt, so the committed
worlds (and everything trained or tested on them) stay as they are.

`find(landmarks, "go to the yellow sphere")` turns a phrase into a landmark, and
`stopping_point(plan, landmark)` gives the spot in front of it that the chair drives to.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from envs.floorplan import FREE, ROUTE_CELL_M

HEIGHT_M = 1.55          # the marker's centre above the floor: about a seated rider's eye height
WALL_GAP_M = 0.25        # how far the marker's centre hangs from the wall
AIM_RANGE_M = 30.0      # the farthest the rider's view reaches: the red line and the markers stop here
REACH_M = 0.8           # a landmark must be within this of floor the chair can get to
PALETTE = {              # colour name: red, green, blue
    "yellow": (1.00, 0.85, 0.05),
    "red": (0.90, 0.10, 0.10),
    "blue": (0.10, 0.30, 0.95),
    "green": (0.10, 0.70, 0.20),
    "orange": (1.00, 0.50, 0.00),
    "purple": (0.60, 0.20, 0.80),
    "pink": (1.00, 0.45, 0.70),
    "cyan": (0.05, 0.80, 0.85),
}
SHAPES = {               # shape name: Webots geometry, about 30 cm across
    "sphere": "Sphere { radius 0.15 subdivision 2 }",
    "cube": "Box { size 0.28 0.28 0.28 }",
    "cone": "Cone { bottomRadius 0.16 height 0.32 }",
    "cylinder": "Cylinder { radius 0.13 height 0.30 }",
}
MAX_LANDMARKS = len(PALETTE)


@dataclass
class Landmark:
    name: str
    colour: str
    shape: str
    x: float
    y: float

    @property
    def rgb(self) -> tuple[float, float, float]:
        return PALETTE[self.colour]


def reachable_area(plan) -> np.ndarray:
    """Pixels within REACH_M of floor the chair can get to: the plan's reachable, roomy route-grid
    cells, grown a little. A landmark must be on one of these."""
    step = max(1, round(ROUTE_CELL_M / plan.resolution))
    reach = np.zeros(plan.cells.shape, np.uint8)
    for row, col in plan.spawn_cells:
        reach[row * step:(row + 1) * step, col * step:(col + 1) * step] = 1
    grow = int(round(REACH_M / plan.resolution)) * 2 + 1
    return cv2.dilate(reach, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))


def ray_distances(plan, origin, angles, max_m: float = AIM_RANGE_M, over_low: bool = True,
                  step_m: float = 0.05) -> np.ndarray:
    """How far a person at `origin` sees along each of `angles` (radians) before a wall or the edge
    of the map: max_m if nothing is in the way. With `over_low` (the default, a rider's eyes)
    low obstacles are looked over; with False it is what the chair's lidar, at 33 cm, meets."""
    angles = np.asarray(angles, dtype=float)
    t = np.arange(step_m, max_m + step_m, step_m)
    xs = origin[0] + np.cos(angles)[:, None] * t[None, :]
    ys = origin[1] + np.sin(angles)[:, None] * t[None, :]
    rows, cols = plan.cells.shape
    c = ((xs + plan.width_m / 2) / plan.resolution).astype(int)
    r = ((plan.height_m / 2 - ys) / plan.resolution).astype(int)
    outside = (c < 0) | (c >= cols) | (r < 0) | (r >= rows)
    c, r = np.clip(c, 0, cols - 1), np.clip(r, 0, rows - 1)
    blocked = (plan.cells[r, c] != FREE) | outside
    low = getattr(plan, "low", None)
    if over_low and low is not None:
        blocked &= ~(low[r, c] & ~outside)
    first = blocked.argmax(axis=1)
    return np.where(blocked.any(axis=1), t[first], max_m)


def choose(plan, n: int, rng: random.Random, origin: tuple[float, float] | None = None, heading: float = 0.0,
           near_m: float = 3.0, far_m: float = 15.0) -> list[Landmark]:
    """`n` landmarks along the walls of `plan`, as far from each other as they can be.

    With an `origin` (where the chair is, and which way it points) they are only where a chair
    there could see them: between near_m and far_m away with nothing in the way, as an
    unfamiliar place shows itself bit by bit. They are numbered from the chair's left to its right."""
    n = max(0, min(n, MAX_LANDMARKS))
    free = (plan.cells == FREE).astype(np.uint8)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * plan.resolution      # metres to the nearest wall

    reach = reachable_area(plan)

    ring = (clearance > WALL_GAP_M - plan.resolution) & (clearance <= WALL_GAP_M + plan.resolution) & (reach > 0)
    rows, cols = np.nonzero(ring)
    if n == 0 or len(rows) == 0:
        return []
    keep = np.arange(0, len(rows), 4)                          # every 4th pixel is plenty
    xs = cols[keep] * plan.resolution + plan.resolution / 2 - plan.width_m / 2
    ys = plan.height_m / 2 - (rows[keep] * plan.resolution + plan.resolution / 2)
    points = np.stack([xs, ys], axis=1)
    if origin is not None:
        points = visible_from(plan, points, origin, near_m, far_m, rng, need=n)
        if len(points) == 0:
            return []

    # Farthest-point picking: the first at random, each next one as far as possible from those so far.
    chosen = [int(rng.randrange(len(points)))]
    gap = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < min(n, len(points)):
        chosen.append(int(gap.argmax()))
        gap = np.minimum(gap, np.linalg.norm(points - points[chosen[-1]], axis=1))

    if origin is not None:     # number them from the chair's left to its right
        bearing = [math.atan2(points[k][1] - origin[1], points[k][0] - origin[0]) - heading for k in chosen]
        bearing = [(b + math.pi) % (2 * math.pi) - math.pi for b in bearing]
        chosen = [k for _, k in sorted(zip(bearing, chosen), reverse=True)]

    # Unique colour and shape pairs, every colour used before any is used twice.
    colours = list(PALETTE)
    rng.shuffle(colours)
    shapes = list(SHAPES)
    rng.shuffle(shapes)
    return [Landmark(f"{colours[i]} {shapes[i % len(shapes)]}", colours[i], shapes[i % len(shapes)],
                     float(points[k][0]), float(points[k][1])) for i, k in enumerate(chosen)]


def visible_from(plan, points: np.ndarray, origin, near_m: float, far_m: float, rng: random.Random,
                 need: int) -> np.ndarray:
    """The `points` between near_m and far_m from `origin` with a clear line to it (no wall or
    furniture between). If there are fewer than `need`, the range is widened a step at a time."""
    from webots.people import line_of_sight
    distance = np.linalg.norm(points - np.array(origin), axis=1)
    for near, far in ((near_m, far_m), (near_m / 2, far_m * 1.5), (1.0, far_m * 3)):
        ring = np.nonzero((distance >= near) & (distance <= far))[0]
        if len(ring) > 300:                                    # looking at every point is slow
            ring = np.array(sorted(rng.sample(list(ring), 300)))
        seen = [i for i in ring if line_of_sight(plan, tuple(origin), tuple(points[i]))]
        if len(seen) >= need:
            return points[seen]
    return points[seen] if seen else points[:0]


def choose_by_view(plan, n: int, rng: random.Random, origin, heading: float, near_m: float = 1.5,
                   far_m: float = AIM_RANGE_M) -> list[Landmark]:
    """Landmarks picked by looking, the same way every time for the same spot and heading: the
    rider's view all the way round is cut into `n` equal sectors (the first centred on straight
    ahead), and in each one a landmark goes on the farthest wall in sight, looking over low
    obstacles. So the farthest wall anywhere is always marked, and so is a long way in every
    other direction there is one. They are numbered from the chair's left to its right."""
    n = max(0, min(n, MAX_LANDMARKS))
    if n == 0:
        return []
    angles = heading + np.linspace(0.0, 2 * math.pi, 180, endpoint=False)
    distance = ray_distances(plan, origin, angles, max_m=far_m)
    spots = np.stack([origin[0] + np.cos(angles) * (distance - WALL_GAP_M),
                      origin[1] + np.sin(angles) * (distance - WALL_GAP_M)], axis=1)
    rows = np.clip(((plan.height_m / 2 - spots[:, 1]) / plan.resolution).astype(int), 0, plan.cells.shape[0] - 1)
    cols = np.clip(((spots[:, 0] + plan.width_m / 2) / plan.resolution).astype(int), 0, plan.cells.shape[1] - 1)
    usable = (distance >= near_m) & (distance < far_m) & (reachable_area(plan)[rows, cols] > 0)
    relative = (angles - heading + math.pi) % (2 * math.pi) - math.pi       # -pi..pi, positive to the left
    sector = (((angles - heading) + math.pi / n) % (2 * math.pi) // (2 * math.pi / n)).astype(int)
    chosen = []
    for k in range(n):
        in_sector = np.nonzero(usable & (sector == k))[0]
        if len(in_sector):
            chosen.append(int(in_sector[distance[in_sector].argmax()]))
    chosen.sort(key=lambda i: -relative[i])                                  # left to right
    colours = list(PALETTE)
    rng.shuffle(colours)
    shapes = list(SHAPES)
    rng.shuffle(shapes)
    return [Landmark(f"{colours[i]} {shapes[i % len(shapes)]}", colours[i], shapes[i % len(shapes)],
                     float(spots[k][0]), float(spots[k][1])) for i, k in enumerate(chosen)]


def node_text(index: int, lm: Landmark) -> str:
    r, g, b = lm.rgb
    colour = f"{r:.2f} {g:.2f} {b:.2f}"
    return f"""DEF LANDMARK{index} Pose {{
  translation {lm.x:.3f} {lm.y:.3f} 0
  children [
    Pose {{ translation 0 0 {HEIGHT_M} children [ Shape {{
      appearance PBRAppearance {{ baseColor {colour} emissiveColor {colour} emissiveIntensity 0.4 roughness 0.6 metalness 0 }}
      geometry {SHAPES[lm.shape]} }} ] }}
    Pose {{ translation 0 0 0.006 children [ Shape {{
      appearance PBRAppearance {{ baseColor {colour} transparency 0.35 roughness 1 metalness 0 }}
      geometry Cylinder {{ height 0.01 radius 0.2 }} }} ] }}
  ]
}}"""


def add_to_world(sup, landmarks: list[Landmark]) -> None:
    """Draw the landmarks in the running world (once; a world that already has them is left alone)."""
    children = sup.getRoot().getField("children")
    for i, lm in enumerate(landmarks):
        if sup.getFromDef(f"LANDMARK{i}") is None:
            children.importMFNodeFromString(-1, node_text(i, lm))


SELECTOR_NODE = """DEF LANDMARK_SELECTOR Pose {
  translation 0 -1000 0
  children [ Pose { translation 0 0 1.2 children [ Shape {
    appearance PBRAppearance { baseColor 1 1 1 emissiveColor 1 1 1 emissiveIntensity 0.5 transparency 0.7 roughness 1 metalness 0 }
    geometry Cylinder { height 2.4 radius 0.32 } } ] } ]
}"""


class Selector:
    """The landmark the rider has picked: a tall, see-through white beam standing on it, easy to
    spot from above and from the first-person view. `select(None)` hides it."""

    def __init__(self, sup, landmarks: list[Landmark]):
        self.landmarks, self.index = landmarks, None
        if sup.getFromDef("LANDMARK_SELECTOR") is None:
            sup.getRoot().getField("children").importMFNodeFromString(-1, SELECTOR_NODE)
        self.position = sup.getFromDef("LANDMARK_SELECTOR").getField("translation")

    @property
    def selected(self) -> Landmark | None:
        return None if self.index is None else self.landmarks[self.index]

    def select(self, index: int | None) -> Landmark | None:
        """Pick landmark number `index` (counting from 0); None, or a number there is no landmark for, clears it."""
        self.index = index if index is not None and 0 <= index < len(self.landmarks) else None
        lm = self.selected
        self.position.setSFVec3f([lm.x, lm.y, 0.0] if lm else [0.0, -1000.0, 0.0])
        return lm


class Landmarks:
    """The landmarks on show, which are chosen afresh around the chair whenever `refresh` is called:
    wherever the rider has got to, they get new places to head for. Redrawn in the running world."""

    def __init__(self, sup, plan, world_name: str, n: int):
        self.sup, self.plan, self.world, self.n = sup, plan, world_name, n
        self.items: list[Landmark] = []
        self.drawn: list[str] = []          # the DEF names of the nodes now in the world
        self.times = 0
        self.selector = Selector(sup, self.items)

    def refresh(self, x: float, y: float, heading: float, seed: int = 0) -> list[Landmark]:
        """Take the old ones down and hang new ones where a chair at (x, y) could see them."""
        for name in self.drawn:
            node = self.sup.getFromDef(name)
            if node is not None:
                node.remove()
        rng = random.Random(f"landmarks {self.world} {seed} {self.times}")
        self.items = choose_by_view(self.plan, self.n, rng, (x, y), heading)
        children = self.sup.getRoot().getField("children")
        self.drawn = []
        for i, lm in enumerate(self.items):
            name = f"LANDMARK{self.times}_{i}"
            children.importMFNodeFromString(-1, node_text(i, lm).replace(f"DEF LANDMARK{i} ", f"DEF {name} ", 1))
            self.drawn.append(name)
        self.selector.landmarks = self.items
        self.selector.select(None)
        self.times += 1
        return self.items

    def take_down(self) -> None:
        """No markers on show (the open world shows them only while the rider drives)."""
        for name in self.drawn:
            node = self.sup.getFromDef(name)
            if node is not None:
                node.remove()
        self.items, self.drawn = [], []
        self.selector.landmarks = self.items
        self.selector.select(None)


LOCK_DEGREES = 5.0       # a landmark this close to straight ahead is locked on
LOCK_SIDEWAYS_M = 0.5    # or this close to the line of sight, for a landmark that is not far
AIM_LASER_NODE = """DEF AIM_LASER Transform {
  translation 0 0 -1000
  children [ Shape {
    appearance PBRAppearance { baseColor 1 0.1 0.1 emissiveColor 1 0.1 0.1 emissiveIntensity 0.8 transparency 0.35 roughness 1 metalness 0 }
    geometry Box { size 1 0.03 0.03 } } ]
}"""
AIM_POST_NODE = """DEF AIM_POST Pose {
  translation 0 0 -1000
  children [ Pose { translation 0 0 0.95 children [ Shape {
    appearance PBRAppearance { baseColor 1 0.1 0.1 emissiveColor 1 0.1 0.1 emissiveIntensity 0.8 transparency 0.3 roughness 1 metalness 0 }
    geometry Cylinder { height 1.9 radius 0.05 } } ] } ]
}"""


class Aim:
    """Where the chair is pointing: a thin red line from the chair (at about the rider's eye height
    for the chair's lidar) along its heading to the first wall in sight, looking over low
    obstacles, with a red post standing where it meets the wall. A landmark within a few degrees of
    that line is "locked on". `update` is called every step while the rider drives, so turning the
    chair sweeps the line across the walls and the markers lock on as they come into line."""

    def __init__(self, sup):
        self.sup = sup
        if sup.getFromDef("AIM_LASER") is None:
            children = sup.getRoot().getField("children")
            for text in (AIM_LASER_NODE, AIM_POST_NODE):
                children.importMFNodeFromString(-1, text)
        self.laser = sup.getFromDef("AIM_LASER")
        self.post = sup.getFromDef("AIM_POST")
        self.shown = False

    def update(self, plan, x: float, y: float, heading: float, landmarks: list[Landmark]):
        """Draw the line from (x, y) along `heading`. Returns (the point it ends at, its length,
        the index of the landmark locked on, or None)."""
        length = float(ray_distances(plan, (x, y), [heading], max_m=AIM_RANGE_M)[0])
        end = (x + math.cos(heading) * length, y + math.sin(heading) * length)
        self.laser.getField("translation").setSFVec3f([(x + end[0]) / 2, (y + end[1]) / 2, 0.9])
        self.laser.getField("rotation").setSFRotation([0.0, 0.0, 1.0, heading])
        self.laser.getField("scale").setSFVec3f([max(length, 0.01), 1.0, 1.0])
        self.post.getField("translation").setSFVec3f(
            [x + math.cos(heading) * (length - 0.1), y + math.sin(heading) * (length - 0.1), 0.0])
        self.shown = True
        return end, length, self.locked(plan, x, y, heading, length, landmarks)

    @staticmethod
    def locked(plan, x, y, heading, length, landmarks) -> int | None:
        from webots.people import line_of_sight
        best, best_angle = None, math.inf
        for i, lm in enumerate(landmarks):
            distance = math.dist((x, y), (lm.x, lm.y))
            angle = abs((math.atan2(lm.y - y, lm.x - x) - heading + math.pi) % (2 * math.pi) - math.pi)
            sideways = distance * math.sin(min(angle, math.pi / 2))
            close = angle <= math.radians(LOCK_DEGREES) or (sideways <= LOCK_SIDEWAYS_M and angle < math.radians(25))
            if close and distance <= length + 1.0 and angle < best_angle \
                    and line_of_sight(plan, (x, y), (lm.x, lm.y), over_low=True):
                best, best_angle = i, angle
        return best

    def hide(self) -> None:
        for node, spot in ((self.laser, [0.0, 0.0, -1000.0]), (self.post, [0.0, 0.0, -1000.0])):
            node.getField("translation").setSFVec3f(spot)
        self.shown = False


def for_world(sup, plan, name: str, n: int) -> list[Landmark]:
    """The landmarks of world `name`, chosen the same way every time, drawn in the running world."""
    landmarks = choose(plan, n, random.Random(f"landmarks {name}"))
    add_to_world(sup, landmarks)
    return landmarks


def stopping_point(plan, lm: Landmark) -> tuple[float, float]:
    """Where the chair should stop for `lm`: the nearest spot it can reach with room around it.
    (A landmark hangs 25 cm from a wall, closer than the chair's centre can get.)"""
    cells = plan.spawn_cells
    centres = np.array([plan.centre_of(int(r), int(c)) for r, c in cells])
    best = int(np.linalg.norm(centres - np.array([lm.x, lm.y]), axis=1).argmin())
    return float(centres[best][0]), float(centres[best][1])


def find(landmarks: list[Landmark], phrase: str) -> Landmark | None:
    """The landmark a phrase names: it needs the shape and, if more than one has that shape, the
    colour too ("the yellow sphere", "go to the sphere" if there is only one). None if the phrase
    names none, or the shape is not enough and no colour is given."""
    words = set(phrase.lower().replace(",", " ").replace(".", " ").split())
    shapes = {s for s in SHAPES if s in words or s + "s" in words} | ({"cube"} if "box" in words else set())
    colours = {c for c in PALETTE if c in words}
    matches = [lm for lm in landmarks
               if (not shapes or lm.shape in shapes) and (not colours or lm.colour in colours)]
    return matches[0] if len(matches) == 1 and (shapes or colours) else None
