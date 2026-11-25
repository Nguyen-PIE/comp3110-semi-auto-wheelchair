"""Low obstacles on the floor (coffee tables, poufs, planter boxes, and a large dining or meeting
table or two) that a rider can see over.

    python webots/run.py fp_hospital --landmarks --low-obstacles      # 4 of them, new ones every trip

A seated rider's eyes are at about 1.15 m, so they look over a 40 cm coffee table at the wall
behind it, and a marker hung there is in plain view. The chair's lidar is at 33 cm, so it does
see the table, and the chair has to drive round it. This module gives the world that:

  * the obstacle is drawn in Webots (a solid box, so the lidar hits it),
  * it is stamped into the floor plan as a wall, so routes go round it, trips that touch it end
    as a wall collision, and people do not walk through it,
  * `plan.low` marks those pixels, so `people.line_of_sight(..., over_low=True)` and the marker
    code can see over them while the chair's own sensors cannot.

They are runtime-only (not saved in the .wbt) and are taken down and put up again for each trip,
so the committed worlds and everything trained on them stay as they are.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from envs.floorplan import FREE, WALL
from webots.make_world import footprint, plan_from_cells, routes_survive

KINDS = {                # kind: (length, depth, height) in metres, colour
    "coffee table": ((1.1, 0.6, 0.42), "0.55 0.40 0.28"),
    "pouf": ((0.6, 0.6, 0.40), "0.75 0.30 0.30"),
    "planter box": ((0.9, 0.4, 0.38), "0.30 0.45 0.25"),
    "dining table": ((2.0, 1.0, 0.75), "0.60 0.45 0.30"),      # large and higher: still below a seated rider's eyes
    "meeting table": ((2.6, 1.1, 0.75), "0.35 0.40 0.50"),
}
LARGE = ("dining table", "meeting table")
LARGE_PER_TRIP = 2       # of the N obstacles, this many are large tables (fewer if N is small)
PASSAGE_M = 1.0          # open floor left around each one, so the chair can always go round
MAX_HEIGHT_M = 0.8       # lower than this, a seated rider (eyes at 1.15 m) looks over it


@dataclass
class LowItem:
    kind: str
    x: float
    y: float
    yaw: float

    @property
    def size(self):
        return KINDS[self.kind][0]


def place(plan, n: int, rng: random.Random) -> tuple[np.ndarray, list[LowItem]]:
    """Up to `n` low obstacles on open floor of `plan`, each with PASSAGE_M of room around it, the
    first few of them large tables. Returns the new cells and the items."""
    cells = plan.cells.copy()
    taken = np.zeros(cells.shape, bool)
    free_px = np.argwhere(plan.clearance > 0.2)
    items: list[LowItem] = []
    if len(free_px) == 0:
        return cells, items
    small = [k for k in KINDS if k not in LARGE]
    for _ in range(n * 120):
        if len(items) >= n:
            break
        kind = rng.choice(LARGE if len(items) < min(LARGE_PER_TRIP, n // 2 or 1) else small)
        length, depth, _ = KINDS[kind][0]
        row, col = free_px[rng.randrange(len(free_px))]
        if float(plan.clearance[row, col]) < math.hypot(length, depth) / 2 + PASSAGE_M:
            continue
        item = LowItem(kind, (col + 0.5) * plan.resolution - plan.width_m / 2,
                       plan.height_m / 2 - (row + 0.5) * plan.resolution,
                       rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.3, 0.3))
        body = footprint(plan, item)
        if (cells[body] != FREE).any() or taken[body].any():
            continue
        cells[body] = WALL
        taken |= footprint(plan, item, grow=PASSAGE_M / 2)
        items.append(item)
    return cells, items


def node_text(i: int, item: LowItem) -> str:
    (length, depth, height), colour = KINDS[item.kind]
    box = f"Box {{ size {length:.2f} {depth:.2f} {height:.2f} }}"
    return f"""DEF LOW{i} Solid {{
  translation {item.x:.3f} {item.y:.3f} {height / 2:.3f}
  rotation 0 0 1 {item.yaw:.4f}
  children [ Shape {{
    appearance PBRAppearance {{ baseColor {colour} roughness 0.8 metalness 0 }}
    geometry {box} }} ]
  name "low obstacle {i}"
  boundingObject {box}
}}"""


class LowObstacles:
    """The low obstacles of the trip on show. `put_up(seed)` takes the last trip's down and puts
    new ones up, in the world and in the floor plan; call it before `scene.reset`, so that the
    trip's start, goal and people are chosen with them in place."""

    def __init__(self, sup, plan, n: int):
        self.sup, self.plan, self.n = sup, plan, n
        self.original = dict(plan.__dict__)         # the plan as it was loaded, to put back
        self.items: list[LowItem] = []
        self.drawn: list[str] = []

    def take_down(self) -> None:
        for name in self.drawn:
            node = self.sup.getFromDef(name)
            if node is not None:
                node.remove()
        self.drawn, self.items = [], []
        self.plan.__dict__.clear()
        self.plan.__dict__.update(self.original)

    def put_up(self, seed: int) -> list[LowItem]:
        self.take_down()
        rng = random.Random(f"low obstacles {self.plan.name} {seed}")
        items: list[LowItem] = []
        for tries in range(4):                      # fewer each time if they cut the building up
            cells, items = place(self.plan, max(0, self.n - tries), rng)
            if not items:
                break
            try:
                new = plan_from_cells(cells, self.plan)
            except ValueError:
                items = []
                continue
            if len(new.spawn_cells) >= 0.85 * len(self.plan.spawn_cells) and routes_survive(self.plan, new, rng):
                low = np.zeros(cells.shape, bool)
                for item in items:
                    low |= footprint(self.plan, item)
                new.low = low
                self.plan.__dict__.update(new.__dict__)
                break
            items = []
        children = self.sup.getRoot().getField("children")
        for i, item in enumerate(items):
            children.importMFNodeFromString(-1, node_text(i, item))
            self.drawn.append(f"LOW{i}")
        self.items = items
        return items
