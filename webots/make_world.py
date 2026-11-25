"""Build a Webots world (.wbt) from one of the 2D simulator's scenarios.

    python webots/make_world.py corridor          # the long, narrow room of s1_corridor
    python webots/make_world.py fp_hospital       # a real floor plan: fp_hospital, fp_office, fp_warehouse
    python webots/make_world.py maps/level2.yaml  # your own floor plan (from scripts/import_floorplan.py)
    python webots/make_world.py all               # every scenario below
    python webots/make_world.py all --variants 3  # ... plus 3 copies of each with random furniture

The 3D world is made from the same floor plan the 2D simulator uses: every wall pixel
of the plan becomes a 2.5 m high wall, so the two simulators have exactly the same
building, in the same coordinates. The world is written to webots/worlds/<name>.wbt,
with a small <name>.json next to it that tells webots/run.py which floor plan to load
(the assistant still plans its routes on the 2D plan, as a real chair would on a map).

The corridor has no floor plan file of its own in the 2D simulator (SocNavGym draws a
new room each episode), so this script draws one: a 22 m by 4.5 m room with two potted
plants, and saves it as an ordinary floor plan in webots/worlds/.

More buildings: fp_house, fp_depot, fp_warehouse2 and fp_warehouse3 are maps from open-source robot
simulations (AWS RoboMaker's small house and small warehouse, MIT-0 licence; Nav2's depot and
warehouse, Apache 2.0). They are downloaded into third_party/floorplans/extra/ the first time.
Areas these maps leave unknown inside the building (shelves the robot that mapped them could not
see into) become solid blocks.

--variants N also writes N furnished copies of each world (<name>_v1, <name>_v2, ...): the same
building with random furniture, so training does not learn only a few fixed layouts. Cabinets,
shelves, benches and beds go against walls, tables and crates in open areas, always leaving a
passage of at least 1.3 m, and a copy is only kept if the chair can still reach most of the
building. The furniture is drawn into the copy's floor plan too (webots/worlds/<name>_vN_map.png),
so routes go round it and touching it counts as hitting a wall, as for the corridor's plants.
With --variants it also writes <name>_test, one more furnished copy that training never uses,
for webots/evaluate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
sys.path.insert(0, str(ROOT))

from envs.floorplan import FREE, ROUTE_CELL_M, UNKNOWN, WALL, _downsample, build_plan, load_floorplan  # noqa: E402

WORLDS = WEBOTS_DIR / "worlds"
WALL_HEIGHT_M = 2.5
WALL_RES_M = 0.10          # walls are built from 10 cm blocks: fewer, bigger boxes than the 5 cm map pixels
PEDESTRIAN_HEIGHT = 1.27   # height of the Pedestrian model's hips above the floor
RIDER_NODE = """DEF RIDER Pedestrian {
  translation 0.0 -1010.0 1.27
  name "rider"
  controller "<none>"
  controllerArgs [ "animated" ]
  shirtColor 0.80 0.45 0.15
  pantsColor 0.16 0.20 0.32
}"""                       # the person sitting in the chair (bridge.py seats them and keeps them there)
PARKED = (0.0, -1000.0)    # where unused pedestrians wait, far outside every map

# Scenarios and how many people walk in them (as in envs/configs/).
SCENARIOS = {
    "corridor": {"map": None, "walking": (5, 10), "standing": (0, 0), "groups": (0, 0),
                 "goal_distance": (12, 22), "time_limit_s": 60},
    "fp_hospital": {"map": "third_party/floorplans/wheelchair-navi/carter_hospital_navigation.yaml"},
    "fp_office": {"map": "third_party/floorplans/wheelchair-navi/carter_office_navigation.yaml"},
    "fp_warehouse": {"map": "third_party/floorplans/wheelchair-navi/carter_warehouse_navigation.yaml"},
}
AWS = "https://raw.githubusercontent.com/aws-robotics/"
NAV2 = "https://raw.githubusercontent.com/ros-navigation/navigation2/main/nav2_bringup/maps/"
EXTRA = "third_party/floorplans/extra"
# name: where to download each file from, and its SHA-256, so a changed file is noticed
DOWNLOADS = {
    "fp_house": {
        "map.yaml": (AWS + "aws-robomaker-small-house-world/ros2/maps/turtlebot3_waffle_pi/map.yaml",
                     "1afd338e3d95a22c83ff05e78f0dd5f4f778efa4f64611c1600961f4d503aaed"),
        "map.pgm": (AWS + "aws-robomaker-small-house-world/ros2/maps/turtlebot3_waffle_pi/map.pgm",
                    "4406c72e26c2ef743c8976406495bebc975327f3e322b57c8344f9076a2fe41c")},
    "fp_depot": {
        "depot.yaml": (NAV2 + "depot.yaml", "fc724e706bfb453a5c4bf513644f754130be27b618e59d75293ead1b7e129bf6"),
        "depot.pgm": (NAV2 + "depot.pgm", "71bbac79cfc2f5db89095182bfa86b29d96b07d4ed3b7726e513ed57afffc851")},
    "fp_warehouse2": {
        "map.yaml": (AWS + "aws-robomaker-small-warehouse-world/ros2/maps/005/map.yaml",
                     "a73d9351e9a5f5e6f95e8d37a90279b2014e3106fe0f3edcfef7afdad4ca9bbc"),
        "map_rotated.png": (AWS + "aws-robomaker-small-warehouse-world/ros2/maps/005/map_rotated.png",
                            "73fe0596f70e5984145284cbe48ff7ab9d8af95c44a6676ff88e77344f8dc574")},
    "fp_warehouse3": {
        "warehouse.yaml": (NAV2 + "warehouse.yaml", "ec00d3d25912028e3debed9ded3d6471f70a3bf0f0d1a3f9470458f3068cc585"),
        "warehouse.pgm": (NAV2 + "warehouse.pgm", "934ce2ca1feacc2216b68142349c6ebb3e46cc7c76fd80a9c2d2a6fb85ba7ff7")},
}
SCENARIOS.update({
    "fp_house": {"map": f"{EXTRA}/fp_house/map.yaml", "walking": (2, 5), "standing": (0, 1), "groups": (0, 1),
                 "goal_distance": (4, 12), "time_limit_s": 60},
    "fp_depot": {"map": f"{EXTRA}/fp_depot/depot.yaml", "goal_distance": (8, 25)},
    "fp_warehouse2": {"map": f"{EXTRA}/fp_warehouse2/map.yaml", "walking": (5, 8), "goal_distance": (6, 18)},
    "fp_warehouse3": {"map": f"{EXTRA}/fp_warehouse3/warehouse.yaml", "walking": (10, 15), "standing": (0, 3),
                      "goal_distance": (10, 30), "time_limit_s": 120},
})

# Furniture for --variants: (length, depth, height) in metres, colour, and whether it stands against a wall.
FURNITURE = {
    "cabinet": ((0.9, 0.45, 1.8), "0.55 0.57 0.60", True),
    "shelf": ((1.2, 0.40, 1.5), "0.60 0.45 0.30", True),
    "bench": ((1.6, 0.45, 0.45), "0.35 0.25 0.20", True),
    "bed": ((2.0, 0.95, 0.6), "0.85 0.87 0.90", True),
    "table": ((1.4, 0.8, 0.75), "0.65 0.50 0.35", False),
    "crate": ((0.8, 0.8, 0.8), "0.70 0.55 0.30", False),
}
PASSAGE_M = 1.3          # room left free in front of and around every piece of furniture
FURNITURE_PER_M2 = 1 / 15

FLOORPLAN_DEFAULTS = {"walking": (8, 12), "standing": (0, 2), "groups": (0, 2),
                      "goal_distance": (10, 25), "time_limit_s": 90, "people_within": 6.0}

SHIRTS = ["0.25 0.55 0.20", "0.70 0.15 0.15", "0.15 0.35 0.70", "0.85 0.65 0.15", "0.55 0.25 0.60",
          "0.90 0.90 0.90", "0.20 0.20 0.20", "0.10 0.55 0.55"]
PANTS = ["0.24 0.25 0.50", "0.15 0.15 0.15", "0.45 0.35 0.25", "0.30 0.30 0.35"]


# ---------------------------------------------------------------- the corridor's floor plan
def draw_corridor(path: Path, length_m=22.0, width_m=4.5, res=0.05) -> list[tuple[float, float, float]]:
    """Draw the corridor as a ROS map (image + YAML) centred on (0, 0). Returns its plants (x, y, r)."""
    wall = 0.2
    border = 0.5
    w_px = int(round((length_m + 2 * (wall + border)) / res))
    h_px = int(round((width_m + 2 * (wall + border)) / res))
    img = np.full((h_px, w_px), 205, np.uint8)           # grey = unknown (outside)

    def px(x, y):
        return int(round((x + w_px * res / 2) / res)), int(round((h_px * res / 2 - y) / res))

    x0, y0 = px(-length_m / 2 - wall, width_m / 2 + wall)
    x1, y1 = px(length_m / 2 + wall, -width_m / 2 - wall)
    img[y0:y1, x0:x1] = 0                                # black = wall
    x0, y0 = px(-length_m / 2, width_m / 2)
    x1, y1 = px(length_m / 2, -width_m / 2)
    img[y0:y1, x0:x1] = 254                              # white = open floor
    plants = [(-3.0, width_m / 2 - 0.5, 0.4), (5.0, -width_m / 2 + 0.5, 0.4)]
    for x, y, r in plants:
        cv2.circle(img, px(x, y), int(round(r / res)), 0, -1)
    cv2.imwrite(str(path.with_suffix(".png")), img)
    origin = [-w_px * res / 2, -h_px * res / 2, 0.0]
    path.with_suffix(".yaml").write_text(yaml.safe_dump({
        "image": path.with_suffix(".png").name, "resolution": res, "origin": origin,
        "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}))
    return plants


# ---------------------------------------------------------------- the extra floor plans
def download(name: str) -> None:
    """Fetch an extra floor plan into third_party/floorplans/extra/<name>/, once."""
    folder = ROOT / EXTRA / name
    for file, (url, sha256) in DOWNLOADS[name].items():
        path = folder / file
        if path.exists():
            continue
        print(f"Downloading {url}")
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != sha256:
            sys.exit(f"{url} has changed since this script was written, so it was not used.")
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def fill_enclosed_unknown(cells: np.ndarray) -> np.ndarray:
    """Unknown areas that do not reach the edge of the map are inside the building: make them solid."""
    unknown = (cells == UNKNOWN).astype(np.uint8)
    n, labels = cv2.connectedComponents(unknown, connectivity=4)
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    cells = cells.copy()
    for k in range(1, n):
        if k not in edge:
            cells[labels == k] = WALL
    return cells


def save_plan(cells: np.ndarray, plan, path: Path) -> None:
    """Write `cells` (the same grid as `plan`) as a ROS map whose frame matches `plan`'s."""
    img = np.full(cells.shape, 205, np.uint8)
    img[cells == FREE] = 254
    img[cells == WALL] = 0
    cv2.imwrite(str(path.with_suffix(".png")), img)
    cx, cy = plan.offset
    path.with_suffix(".yaml").write_text(yaml.safe_dump({
        "image": path.with_suffix(".png").name, "resolution": plan.resolution,
        "origin": [float(cx - plan.width_m / 2), float(cy - plan.height_m / 2), 0.0],
        "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}))


def to_route_grid_resolution(cells: np.ndarray, plan, res: float = 0.05):
    """Resample a map whose pixel size does not divide the route grid (envs/floorplan.py's
    ROUTE_CELL_M, 0.25 m) to 5 cm pixels. With 3 cm pixels the route grid's cells come out 0.24 m
    instead of 0.25 m, so start and goal points drift away from the spots that were checked for
    room, by up to a metre across a big map, and trips can start touching a wall."""
    k = ROUTE_CELL_M / plan.resolution
    if abs(k - round(k)) < 1e-6:
        return cells, plan
    size = (int(round(plan.width_m / res)), int(round(plan.height_m / res)))   # (width, height) for cv2
    share = lambda mask: cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    wall, unknown = share(cells == WALL), share(cells == UNKNOWN)
    out = np.full((size[1], size[0]), FREE, np.uint8)
    out[unknown > 0.5] = UNKNOWN
    out[wall > 0.2] = WALL                     # keep thin walls
    return out, build_plan(plan.name, res, out == WALL, out == FREE, plan.offset)


def plan_from_cells(cells: np.ndarray, plan):
    return build_plan(plan.name, plan.resolution, cells == WALL, cells == FREE, plan.offset)


# ---------------------------------------------------------------- random furniture
@dataclass
class Item:
    kind: str
    x: float
    y: float
    yaw: float

    @property
    def size(self):
        return FURNITURE[self.kind][0]


def footprint(plan, item: Item, grow: float = 0.0) -> np.ndarray:
    """Pixels of `plan` covered by the item (enlarged by `grow` metres on every side)."""
    length, depth, _ = item.size
    c, s = math.cos(item.yaw), math.sin(item.yaw)
    corners = []
    for u, v in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        dx, dy = u * (length / 2 + grow), v * (depth / 2 + grow)
        x, y = item.x + c * dx - s * dy, item.y + s * dx + c * dy
        corners.append(((x + plan.width_m / 2) / plan.resolution, (plan.height_m / 2 - y) / plan.resolution))
    mask = np.zeros(plan.cells.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(corners) * 8).astype(np.int32)], 1, shift=3)
    return mask.astype(bool)


def place_furniture(plan, n: int, rng: random.Random) -> tuple[np.ndarray, list[Item]]:
    """Up to n pieces of furniture on `plan`'s open floor. Returns the new cells and the items."""
    cells = plan.cells.copy()
    near_furniture = np.zeros(cells.shape, bool)       # keeps pieces PASSAGE_M apart
    free_px = np.argwhere(plan.clearance > 0.2)
    items: list[Item] = []
    kinds = list(FURNITURE)
    for _ in range(n * 40):
        if len(items) >= n:
            break
        kind = rng.choice(kinds)
        (length, depth, _), _, against_wall = FURNITURE[kind]
        row, col = free_px[rng.randrange(len(free_px))]
        clear = float(plan.clearance[row, col])
        x = (col + 0.5) * plan.resolution - plan.width_m / 2
        y = plan.height_m / 2 - (row + 0.5) * plan.resolution
        if against_wall:
            if not depth / 2 + 0.03 <= clear <= depth / 2 + 0.12:
                continue
            nx, ny = plan.away_from_wall(x, y)
            if nx == ny == 0:
                continue
            ahead = (x + nx * (depth / 2 + PASSAGE_M), y + ny * (depth / 2 + PASSAGE_M))
            if plan.clearance_at(*ahead) < 0.5:                         # keep the corridor passable
                continue
            item = Item(kind, x, y, math.atan2(ny, nx) + math.pi / 2)    # long side along the wall
        else:
            if clear < math.hypot(length, depth) / 2 + PASSAGE_M:
                continue
            item = Item(kind, x, y, rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.2, 0.2))
        body = footprint(plan, item)
        if (cells[body] != FREE).mean() > (0.05 if against_wall else 0.0) or near_furniture[body].any():
            continue
        cells[body] = WALL
        near_furniture |= footprint(plan, item, grow=PASSAGE_M / 2)
        items.append(item)
    return cells, items


def furnished(plan, rng: random.Random, tries: int = 6):
    """A copy of `plan` with random furniture that still lets the chair get around."""
    free_m2 = float((plan.cells == FREE).sum()) * plan.resolution ** 2
    n = min(60, max(4, int(free_m2 * FURNITURE_PER_M2)))
    for _ in range(tries):
        cells, items = place_furniture(plan, n, rng)
        try:
            new = plan_from_cells(cells, plan)
        except ValueError:
            n = int(n * 0.7)
            continue
        if len(new.spawn_cells) >= 0.6 * len(plan.spawn_cells) and routes_survive(plan, new, rng):
            return cells, items
        n = int(n * 0.7)
    return plan.cells.copy(), []


def routes_survive(old, new, rng: random.Random, pairs: int = 20) -> bool:
    """Whether trips that were possible before the furniture still are (nearly all of them)."""
    failed = 0
    for _ in range(pairs):
        a, b = new.random_point(rng), new.random_point(rng)
        if old.route(a, b) is not None and new.route(a, b) is None:
            failed += 1
    return failed <= pairs // 10


def furniture_nodes(items: list[Item]) -> tuple[str, str]:
    """The furniture as shapes to draw and boxes for collisions, both added to the walls' Solids.
    (A separate Solid per piece made Webots test each piece against every wall block at every
    physics step, which slowed big buildings down more than ten times.)"""
    shapes, boxes = [], []
    for it in items:
        (length, depth, height), colour, _ = FURNITURE[it.kind]
        box = f"Box {{ size {length:.2f} {depth:.2f} {height:.2f} }}"
        where = f"translation {it.x:.3f} {it.y:.3f} {height / 2:.3f} rotation 0 0 1 {it.yaw:.4f}"
        shapes.append(f"    Pose {{ {where} children [ Shape {{ appearance PBRAppearance {{ "
                      f"baseColor {colour} roughness 0.8 metalness 0 }} geometry {box} }} ] }}  # {it.kind}")
        boxes.append(f"        Pose {{ {where} children [ {box} ] }}")
    return "\n".join(shapes), "\n".join(boxes)


# ---------------------------------------------------------------- walls
def wall_rectangles(plan, plants=(), res: float = WALL_RES_M, items=()) -> list[tuple[float, float, float, float]]:
    """The plan's walls as a list of rectangles (x_min, y_min, x_max, y_max) in metres.
    Wall pixels are grouped into res-sized blocks, then runs of blocks in each row are
    merged, and runs with the same span in consecutive rows are merged again. Plants are
    on the plan too (so routes go round them) but are left out here: they get their own model.
    So is furniture (`items`)."""
    walls = plan.cells == WALL
    for item in items:
        walls &= ~footprint(plan, item, grow=2 * plan.resolution)
    rows, cols = np.indices(walls.shape)
    for x, y, r in plants:
        px = (x + plan.width_m / 2) / plan.resolution
        py = (plan.height_m / 2 - y) / plan.resolution
        walls &= (cols - px) ** 2 + (rows - py) ** 2 > (r / plan.resolution + 1.5) ** 2
    k = max(1, int(round(res / plan.resolution)))
    res = k * plan.resolution
    mask = _downsample(walls, k, "any")
    rects, open_runs = [], {}      # (first col, last col) -> first row of the rectangle
    for r in range(mask.shape[0] + 1):
        runs = set()
        if r < mask.shape[0]:
            row = np.concatenate([[False], mask[r], [False]])
            edges = np.flatnonzero(row[1:] != row[:-1])
            runs = {(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])}
        for run, r0 in list(open_runs.items()):
            if run not in runs:
                rects.append((r0, r - 1, run[0], run[1]))
                del open_runs[run]
        for run in runs:
            open_runs.setdefault(run, r)
    half_w, half_h = plan.width_m / 2, plan.height_m / 2
    return [(c0 * res - half_w, half_h - (r1 + 1) * res, (c1 + 1) * res - half_w, half_h - r0 * res)
            for r0, r1, c0, c1 in rects]


N_OBJECT_SLOTS = 4   # event objects a world can show at once (bridge.py)
TILE_M = 8.0   # walls are split into Solids of about this size, so the physics only checks nearby walls


def walls_nodes(rects, items=()) -> str:
    """The walls and furniture, as one Solid per TILE_M x TILE_M square of the building. With one
    Solid for a whole hospital, Webots checked the chair against all its ~1200 wall blocks at every
    physics step; split up, it only checks the squares the chair is near."""
    tiles: dict = {}
    for r in rects:
        key = (math.floor((r[0] + r[2]) / 2 / TILE_M), math.floor((r[1] + r[3]) / 2 / TILE_M))
        tiles.setdefault(key, ([], []))[0].append(r)
    for it in items:
        tiles.setdefault((math.floor(it.x / TILE_M), math.floor(it.y / TILE_M)), ([], []))[1].append(it)
    return "\n".join(walls_node(r, items=i, name=f"walls {k}")
                     for k, (r, i) in enumerate(tiles[key] for key in sorted(tiles)))


def walls_node(rects, height=WALL_HEIGHT_M, items=(), name="walls") -> str:
    """One Solid holding every wall: a single mesh to draw (fast) and a box per rectangle for collisions,
    plus the furniture."""
    coords, faces = [], []
    for x0, y0, x1, y1 in rects:
        base = len(coords)
        coords += [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
                   (x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)]
        for f in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
            faces.append(" ".join(str(base + i) for i in f) + " -1")
    boxes = "\n".join(
        f"        Pose {{ translation {(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} {height / 2:.3f} "
        f"children [ Box {{ size {x1 - x0:.3f} {y1 - y0:.3f} {height:.3f} }} ] }}"
        for x0, y0, x1, y1 in rects)
    furniture_shapes, furniture_boxes = furniture_nodes(list(items))
    if furniture_boxes:
        boxes += "\n" + furniture_boxes
    points = ", ".join(f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in coords)
    wall_shape = f"""    Shape {{
      appearance PBRAppearance {{ baseColor 0.85 0.84 0.80 roughness 0.9 metalness 0 }}
      geometry IndexedFaceSet {{
        coord Coordinate {{ point [ {points} ] }}
        coordIndex [ {" ".join(faces)} ]
        creaseAngle 0.5
      }}
    }}""" if rects else ""                     # a square with only furniture has no wall mesh
    return f"""Solid {{
  name "{name}"
  children [
{wall_shape}
{furniture_shapes}
  ]
  boundingObject Group {{
    children [
{boxes}
    ]
  }}
  locked TRUE
}}"""


def plant_node(i, x, y, r) -> str:
    return f"""Solid {{
  translation {x:.3f} {y:.3f} 0
  name "plant {i}"
  children [
    Pose {{ translation 0 0 0.25 children [ Shape {{
      appearance PBRAppearance {{ baseColor 0.55 0.30 0.15 roughness 0.9 metalness 0 }}
      geometry Cylinder {{ height 0.5 radius {r * 0.75:.2f} }} }} ] }}
    Pose {{ translation 0 0 0.85 children [ Shape {{
      appearance PBRAppearance {{ baseColor 0.15 0.50 0.15 roughness 0.9 metalness 0 }}
      geometry Sphere {{ radius {r:.2f} subdivision 2 }} }} ] }}
  ]
  boundingObject Pose {{ translation 0 0 0.6 children [ Cylinder {{ height 1.2 radius {r:.2f} }} ] }}
  locked TRUE
}}"""


def look_at(eye, target) -> str:
    """Viewpoint orientation (axis-angle) for a camera at `eye` looking at `target` (Webots
    viewpoints look along their x axis, with z up)."""
    d = np.subtract(target, eye).astype(float)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))      # positive = looking down
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    m = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    angle = math.acos(np.clip((np.trace(m) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return "0 0 1 0"
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * math.sin(angle))
    return f"{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f} {angle:.4f}"


def world_text(name, plan, plants, n_people, start, items=()) -> str:
    rects = wall_rectangles(plan, plants, items=items)
    sx, sy = start
    eye = (sx - 6.0, sy - 6.0, 7.0)
    pedestrians = "\n".join(
        f"""DEF PED{i} Pedestrian {{
  translation {PARKED[0] + 2 * i:.1f} {PARKED[1]:.1f} {PEDESTRIAN_HEIGHT}
  name "pedestrian {i}"
  controller "<none>"
  controllerArgs [ "animated" ]
  shirtColor {SHIRTS[i % len(SHIRTS)]}
  pantsColor {PANTS[i % len(PANTS)]}
}}""" for i in range(n_people))
    # Moving or out-of-place objects for events (bridge.py sets their size, colour and place):
    # a pet, a child, an emergency trolley, a forklift, a pallet of boxes. Parked out of sight.
    objects = "\n".join(
        f"""DEF OBJ{i} Pose {{
  translation {PARKED[0] + 2 * i:.1f} {PARKED[1] - 20:.1f} 0.5
  children [ Shape {{ appearance DEF OBJ{i}_LOOK PBRAppearance {{ baseColor 0.8 0.5 0.1 roughness 0.7 metalness 0 }}
    geometry DEF OBJ{i}_BOX Box {{ size 0.5 0.5 0.5 }} }} ]
}}""" for i in range(N_OBJECT_SLOTS))
    # The camera follows the chair with followSmoothness 0, exactly: Webots' smoothed following
    # drifts away when the chair jumps back to its start.
    return f"""#VRML_SIM R2025a utf8
# Generated by webots/make_world.py from the {name} floor plan. Re-run it rather than editing.

EXTERNPROTO "../protos/Wheelchair.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/humans/pedestrian/protos/Pedestrian.proto"

WorldInfo {{
  info [ "G19 wheelchair: {name}" ]
  title "G19 wheelchair: {name}"
  basicTimeStep 20
  contactProperties [
    ContactProperties {{ coulombFriction [ 1 ] softCFM 1e-05 }}
    ContactProperties {{ material2 "caster" coulombFriction [ 0 ] softCFM 1e-05 }}
  ]
}}
Viewpoint {{
  orientation {look_at(eye, (sx, sy, 0.0))}
  position {eye[0]:.2f} {eye[1]:.2f} {eye[2]:.2f}
  follow "wheelchair"
  followType "Tracking Shot"
  followSmoothness 0
}}
Background {{ skyColor [ 0.75 0.82 0.90 ] }}
DirectionalLight {{ direction -0.4 -0.3 -1 intensity 2.5 castShadows FALSE }}
DirectionalLight {{ direction 0.3 0.5 -1 intensity 1.0 castShadows FALSE }}
DEF FLOOR Solid {{
  translation 0 0 -0.05
  name "floor"
  children [ Shape {{
    appearance PBRAppearance {{ baseColor 0.62 0.60 0.56 roughness 1 metalness 0 }}
    geometry Box {{ size {plan.width_m:.2f} {plan.height_m:.2f} 0.1 }} }} ]
  boundingObject Box {{ size {plan.width_m:.2f} {plan.height_m:.2f} 0.1 }}
  locked TRUE
}}
{walls_nodes(rects, items)}
{chr(10).join(plant_node(i, *p) for i, p in enumerate(plants))}
DEF GOAL Pose {{
  translation {sx + 3:.2f} {sy:.2f} 0.01
  children [ Shape {{
    appearance PBRAppearance {{ baseColor 0.1 0.8 0.2 transparency 0.5 roughness 1 metalness 0 }}
    geometry Cylinder {{ height 0.02 radius 0.5 }} }} ]
}}
DEF CHAIR Wheelchair {{
  translation {sx:.2f} {sy:.2f} 0
}}
{pedestrians}
{objects}
{RIDER_NODE}
"""


def make(name: str, settings: dict, variant: int | str = 0) -> None:
    """Write worlds/<name>.wbt and <name>.json. With a variant, a furnished copy: <name>_v<variant>,
    or <name>_test for variant "test" (a layout kept for testing, never trained on)."""
    settings = dict(settings)
    plants = []
    if settings["map"] is None:                    # the corridor: draw its floor plan
        map_path = WORLDS / f"{name}_map.yaml"
        plants = draw_corridor(map_path)
        settings["map"] = map_path.relative_to(ROOT).as_posix()
    else:
        settings = {**FLOORPLAN_DEFAULTS, **settings}
    if name in DOWNLOADS:
        download(name)
    map_file = (ROOT / settings["map"]).resolve()
    if not map_file.exists():
        sys.exit(f"Floor plan not found: {map_file}\nRun `bash scripts/get_maps.sh` first for the fp_* scenarios.")
    plan = load_floorplan(str(map_file))

    cells, items = plan.cells, []
    if name in DOWNLOADS:
        cells = fill_enclosed_unknown(cells)
        cells, plan = to_route_grid_resolution(cells, plan)
    if variant:
        base = plan_from_cells(cells, plan) if cells is not plan.cells else plan
        cells, items = furnished(base, random.Random(f"{name} {variant}"))
        name = f"{name}_test" if variant == "test" else f"{name}_v{variant}"
    if cells is not plan.cells:                    # the plan changed: save it next to the world
        map_path = WORLDS / f"{name}_map.yaml"
        save_plan(cells, plan, map_path)
        settings["map"] = map_path.relative_to(ROOT).as_posix()
        plan = load_floorplan(str(map_path))

    # Room for half as many walkers again (training sometimes makes crowds busier: bridge.py, Variety),
    # and for the people of events (people stepping out of rooms, a meeting ending, a waiting area).
    n_people = math.ceil(1.5 * settings["walking"][1]) + settings["standing"][1] + 3 * settings["groups"][1] + 10
    start = plan.random_point(random.Random(0))
    (WORLDS / f"{name}.wbt").write_text(world_text(name, plan, plants, n_people, start, items))
    (WORLDS / f"{name}.json").write_text(json.dumps({**settings, "people_slots": n_people}, indent=2) + "\n")
    print(f"Wrote {WORLDS / (name + '.wbt')}: {plan.width_m:.0f} x {plan.height_m:.0f} m, "
          f"{len(wall_rectangles(plan, plants, items=items))} wall blocks, {len(items)} pieces of furniture, "
          f"{n_people} pedestrians")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenarios", nargs="+", help=f"{', '.join(SCENARIOS)}, all, or a floor-plan YAML path")
    p.add_argument("--variants", type=int, default=0, help="also write this many copies with random furniture")
    args = p.parse_args(argv)
    WORLDS.mkdir(exist_ok=True)

    names = list(SCENARIOS) if args.scenarios == ["all"] else args.scenarios
    for scenario in names:
        if scenario in SCENARIOS:
            name, settings = scenario, SCENARIOS[scenario]
        else:
            name, settings = Path(scenario).stem, {"map": scenario}
        for variant in range(args.variants + 1):
            make(name, settings, variant)
        if args.variants:
            make(name, settings, "test")


if __name__ == "__main__":
    main()
