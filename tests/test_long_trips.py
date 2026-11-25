"""Checks for what keeps a long open-world trip from slowing down: where a sign is, worked out again
only when it is seen again (vision.Seen), the places read, worked out again only when something new is
read (SignReader.places), and the blue line drawn in short pieces (open_world.Trail). No simulator.

    python tests/test_long_trips.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import open_world, vision  # noqa: E402


def seen(tag, x, y, name, kind="plate", times=3):
    s = vision.Seen(tag, kind, name=name, votes={name: 3})
    for k in range(times):
        s.xs.append(x + 0.01 * k), s.ys.append(y), s.zs.append(1.5), s.yaws.append(0.0), s.weights.append(1.0)
    return s


def test_a_sign_is_placed_again_only_when_seen_again():
    s = seen(1, 2.0, 3.0, "Pharmacy")
    first = (s.x, s.y, s.yaw)
    assert math.isclose(first[0], 2.01) and s._at[0] == 3
    s.xs.append(5.0), s.ys.append(3.0), s.zs.append(1.5), s.yaws.append(0.0), s.weights.append(1.0)
    assert s.x > first[0] and s._at[0] == 4          # a new sighting moves it


def test_the_places_read_are_worked_out_again_only_after_something_new():
    reader = vision.SignReader.__new__(vision.SignReader)
    reader.seen = {1: seen(1, 2.0, 3.0, "Pharmacy"), 2: seen(2, 10.0, 3.0, "Cafe")}
    reader.map_plan = None
    reader.scene = SimpleNamespace(plan=None)
    first = reader.places()
    assert set(first) == {"Pharmacy", "Cafe"}
    first["Pharmacy"] = (0, 0)                        # a copy: changing it changes nothing kept
    assert reader.places()["Pharmacy"] != (0, 0)
    reader.seen[3] = seen(3, 20.0, 3.0, "Toilets")
    assert "Toilets" in reader.places()


class FakeField:
    def __init__(self, values):
        self.values = list(values)
        self.inserted = 0

    def setMFVec3f(self, i, v):
        self.values[i] = list(v)

    def setMFInt32(self, i, v):
        self.values[i] = v

    def insertMFVec3f(self, i, v):                     # makes Python wait for Webots: Trail must not use it
        self.inserted += 1

    insertMFInt32 = insertMFVec3f


class FakeSup:
    """Enough of Webots' supervisor for Trail: nodes added by text, found by DEF name, removed."""

    def __init__(self):
        self.nodes = {}

    def getRoot(self):
        sup = self

        class Children:
            def importMFNodeFromString(self, i, text):
                name = text.split()[1]
                points = FakeField([[float(v) for v in p.split()]
                                    for p in text.split("point [ ")[1].split(" ]")[0].split(", ")])
                index = FakeField(int(k) for k in text.split("coordIndex [ ")[1].split(" ]")[0].split())
                line = SimpleNamespace(getField=lambda f: SimpleNamespace(getSFNode=lambda: SimpleNamespace(
                    getField=lambda g: SimpleNamespace(getSFNode=lambda: SimpleNamespace(
                        getField=lambda h: points)) if g == "coord" else index)))
                sup.nodes[name] = SimpleNamespace(remove=lambda: [sup.nodes.pop(name), sup.nodes.pop(f"{name}_LINE")],
                                                  points=points, index=index)   # its line goes with it
                sup.nodes[f"{name}_LINE"] = line
        return SimpleNamespace(getField=lambda f: Children())

    def getFromDef(self, name):
        return self.nodes.get(name)


def test_the_trail_is_drawn_in_short_pieces():
    sup = FakeSup()
    scene = SimpleNamespace(sup=sup, _screen_only=[], only_on_screen=lambda node: None)
    trail = open_world.Trail(scene)
    for k in range(250):
        trail.add(k * 0.1, 0.0)
    assert len(trail.pieces) == 3 and trail.drawn == 250          # 99 new points a piece
    first, second, third = (sup.getFromDef(p) for p in trail.pieces)
    assert len(first.points.values) == len(third.points.values) == open_world.TRAIL_PIECE   # made whole
    assert first.index.values == list(range(open_world.TRAIL_PIECE))                     # all joined
    assert second.points.values[0][:2] == first.points.values[-1][:2]    # where the last piece ended
    joined = [k for k in third.index.values if k >= 0]
    assert joined == list(range(53)) and third.index.values[53:] == [-1] * (open_world.TRAIL_PIECE - 53)
    fresh = open_world.Trail(SimpleNamespace(sup=FakeSup(), _screen_only=[], only_on_screen=lambda node: None))
    fresh.add(1.0, 2.0)                                 # Webots wants two items in a new line's coordIndex
    assert fresh.index.values[:3] == [0, 1, -1]
    assert all(sup.getFromDef(p).points.inserted == sup.getFromDef(p).index.inserted == 0 for p in trail.pieces)
    trail.add(24.92, 0.0)                              # less than 5 cm on: no new point
    assert trail.drawn == 250
    trail.clear()
    assert trail.pieces == [] and trail.drawn == 0 and not any(n.startswith("OPEN_TRAIL") for n in sup.nodes)


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
