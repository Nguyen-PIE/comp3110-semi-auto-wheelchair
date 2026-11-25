"""Checks for the open world's markers (webots/open_world.py BuildingMarkers): put up along the walls
of the whole building, each with its own colour and shape, some always in sight, and listed (the ones
the keys, the red line and voice pick from) as the chair moves. No simulator needed.

    python tests/test_open_world_markers.py          (or: python -m pytest tests, if pytest is installed)
"""

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.floorplan import ROUTE_CELL_M, load_floorplan  # noqa: E402
from webots import landmarks, open_world, signs  # noqa: E402

HOSPITAL = load_floorplan(str(ROOT / json.loads((ROOT / "webots/worlds/fp_hospital.json").read_text())["map"]))
SIGNS_AT = [(f.x, f.y) for f in signs.make_layout(HOSPITAL, "fp_hospital").faces]


def markers_for(plan, world):
    building = open_world.BuildingMarkers.__new__(open_world.BuildingMarkers)   # without a simulator
    building.items, building.scene = open_world.building_markers(plan, world, SIGNS_AT), SimpleNamespace(plan=plan)
    return building


def spots(plan, n, seed=1):
    rng, cells = random.Random(seed), list(plan.spawn_cells)
    for _ in range(n):
        row, col = rng.choice(cells)
        yield (col + 0.5) * ROUTE_CELL_M - plan.width_m / 2, plan.height_m / 2 - (row + 0.5) * ROUTE_CELL_M


def test_the_whole_building_has_markers_each_named_once():
    items = open_world.building_markers(HOSPITAL, "fp_hospital")
    assert len(items) >= 30
    assert len({lm.name for lm in items}) == len(items)
    assert all(lm.colour in landmarks.PALETTE and lm.shape in open_world.MARKER_SHAPES for lm in items)
    assert [lm.name for lm in items] == [lm.name for lm in open_world.building_markers(HOSPITAL, "fp_hospital")]


def test_markers_are_spread_out_along_the_walls():
    items = open_world.building_markers(HOSPITAL, "fp_hospital")
    closest = min(math.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(items) for b in items[i + 1:])
    assert closest >= open_world.MARKER_SPACING_M - 0.01
    reach = landmarks.reachable_area(HOSPITAL)
    for lm in items:
        row = int((HOSPITAL.height_m / 2 - lm.y) / HOSPITAL.resolution)
        col = int((lm.x + HOSPITAL.width_m / 2) / HOSPITAL.resolution)
        assert reach[row, col], lm.name


def test_no_marker_hides_a_sign():
    for lm in open_world.building_markers(HOSPITAL, "fp_hospital", SIGNS_AT):
        assert min(math.hypot(lm.x - x, lm.y - y) for x, y in SIGNS_AT) >= open_world.MARKER_SIGN_GAP_M, lm.name


def test_some_are_always_in_sight():
    building = markers_for(HOSPITAL, "fp_hospital")
    counts = []
    for x, y in spots(HOSPITAL, 60):
        shown = building.in_sight(x, y, 0.3)
        assert len(shown) <= open_world.MARKERS_LISTED
        counts.append(len(shown))
        bearings = [(math.atan2(lm.y - y, lm.x - x) - 0.3 + math.pi) % (2 * math.pi) - math.pi for lm in shown]
        assert bearings == sorted(bearings, reverse=True)       # numbered from the chair's left to its right
    assert min(counts) >= 1 and sum(counts) / len(counts) >= 3


def test_a_marker_is_drawn_with_its_shape():
    lm = landmarks.Landmark("red pyramid", "red", "pyramid", 1.0, 2.0)
    text = open_world.marker_node("OPEN_MARKER0", lm, 1.55)
    assert text.startswith("DEF OPEN_MARKER0 Pose") and "subdivision 4" in text and "translation 1.000 2.000 0" in text


class Selector:
    """landmarks.Selector without the beam in the world."""

    def __init__(self):
        self.landmarks, self.index = [], None

    @property
    def selected(self):
        return None if self.index is None else self.landmarks[self.index]

    def select(self, i):
        self.index = i if i is not None and 0 <= i < len(self.landmarks) else None
        return self.selected


def test_the_listed_markers_follow_the_chair_and_keep_the_one_picked():
    from webots import run
    building = markers_for(HOSPITAL, "fp_hospital")
    x, y = next(spots(HOSPITAL, 1))
    robot = SimpleNamespace(x=x, y=y, orientation=0.0)
    selector = Selector()
    marks = SimpleNamespace(items=[], selector=selector)
    run.list_markers(marks, building, SimpleNamespace(robot=robot))
    assert marks.items and marks.items == building.in_sight(x, y, 0.0)
    picked = marks.items[-1]
    selector.select(len(marks.items) - 1)
    robot.orientation = 1.0                                     # turned: numbered again, still picked
    run.list_markers(marks, building, SimpleNamespace(robot=robot))
    assert selector.selected is not None and selector.selected.name == picked.name


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "-", repr(exc))
    sys.exit(1 if failed else 0)
