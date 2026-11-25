"""Checks for the rider driving (O in webots/run.py): the warning and slowing down for people, walls
and obstacles in the way (webots/guard.py), and the status text sent to Webots at most every
TEXT_EVERY_S. No simulator.

    python tests/test_override.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import guard  # noqa: E402
from webots.bridge import CHAIR_RADIUS_M, PERSON_DIAMETER_M  # noqa: E402


def person(x, y, width=PERSON_DIAMETER_M):
    return SimpleNamespace(x=x, y=y, width=width)


def test_only_someone_straight_ahead_is_in_the_way():
    ahead, beside, behind = person(2.0, 0.0), person(0.5, 1.5), person(-1.5, 0.0)
    gap, who = guard.in_the_way(0.0, 0.0, 0.0, [beside, behind, ahead], CHAIR_RADIUS_M)
    assert who is ahead and math.isclose(gap, 2.0 - CHAIR_RADIUS_M - PERSON_DIAMETER_M / 2)
    gap, who = guard.in_the_way(0.0, 0.0, math.pi / 2, [beside, behind, ahead], CHAIR_RADIUS_M)
    assert who is beside                                 # turned towards them
    assert guard.in_the_way(0.0, 0.0, math.pi, [ahead], CHAIR_RADIUS_M) == (math.inf, None)


def test_a_wider_thing_is_nearer():
    trolley = person(2.0, 0.0, width=1.2)
    gap, _ = guard.in_the_way(0.0, 0.0, 0.0, [trolley], CHAIR_RADIUS_M)
    assert math.isclose(gap, 2.0 - CHAIR_RADIUS_M - 0.6)


def test_speed_goes_down_with_the_gap():
    assert guard.allowed(math.inf) == 1.0 and guard.allowed(guard.SLOW_FROM_M) == 1.0
    assert guard.allowed(guard.STOP_M) == 0.0 and guard.allowed(-0.2) == 0.0
    middle = guard.allowed((guard.SLOW_FROM_M + guard.STOP_M) / 2)
    assert math.isclose(middle, 0.5)


class FakeField:
    def __init__(self):
        self.value = None

    def setSFVec3f(self, v):
        self.value = list(v)


class FakeSup:
    def __init__(self):
        self.ring = FakeField()
        self.nodes = {}

    def getFromDef(self, name):
        return self.nodes.get(name)

    def getRoot(self):
        sup = self
        return SimpleNamespace(getField=lambda f: SimpleNamespace(
            importMFNodeFromString=lambda i, text: sup.nodes.update(
                GUARD_RING=SimpleNamespace(getField=lambda g: sup.ring))))


def scene_with(people, x=0.0, heading=0.0):
    robot = SimpleNamespace(x=x, y=0.0, orientation=heading)
    return SimpleNamespace(sup=FakeSup(), robot=robot, crowd=SimpleNamespace(everyone=people),
                           lidar=SimpleNamespace(getMaxRange=lambda: 4.0), plan=None,
                           only_on_screen=lambda node: None)


def guard_for(scene, brake=True):
    g = guard.Guard.__new__(guard.Guard)
    g.scene, g.brake, g.line_of_sight = scene, brake, lambda plan, a, b: True
    scene.sup.getRoot().getField("children").importMFNodeFromString(-1, guard.RING_NODE)
    g.ring = scene.sup.getFromDef("GUARD_RING").getField("translation")
    g.warning, g.hits, g.touching, g.ring_at = "", 0, False, None
    g.wall_hits, g.touching_wall = 0, False
    return g


def lidar_image(points, n=360, far=4.0):
    """A 360 ray range image (Webots' order: from straight behind, clockwise) that sees `points`,
    (x, y) in the chair's frame, each on the ray nearest its bearing; inf elsewhere."""
    image = [math.inf] * n
    for x, y in points:
        bearing = math.atan2(y, x)                       # anticlockwise from straight ahead
        i = round((math.pi - bearing) / (2 * math.pi / n)) % n
        image[i] = min(image[i], math.hypot(x, y)) if math.hypot(x, y) <= far else image[i]
    return image


def wall_across(distance, half_width=2.0, step=0.01):
    return [(distance, -half_width + k * step) for k in range(int(2 * half_width / step) + 1)]


def wall_along(side, length=4.0, step=0.01):
    return [(-length / 2 + k * step, side) for k in range(int(length / step) + 1)]


def test_a_wall_straight_ahead_is_seen_from_the_front_of_the_chair():
    gap = guard.wall_ahead(lidar_image(wall_across(1.0)), CHAIR_RADIUS_M)
    assert math.isclose(gap, 1.0 - CHAIR_RADIUS_M, abs_tol=0.02)


def test_corridor_walls_and_a_doorway_the_chair_fits_through_are_not_in_the_way():
    corridor = wall_along(0.6) + wall_along(-0.6)
    assert guard.wall_ahead(lidar_image(corridor), CHAIR_RADIUS_M) == math.inf
    frame = [(0.5, 0.45), (0.5, -0.45), (0.55, 0.47), (0.55, -0.47)]   # a 0.9 m doorway, chair in the middle
    assert guard.wall_ahead(lidar_image(frame), CHAIR_RADIUS_M) == math.inf
    assert guard.wall_ahead(lidar_image(wall_across(1.0)), CHAIR_RADIUS_M) < math.inf
    assert guard.wall_ahead([], CHAIR_RADIUS_M) == math.inf


def test_speed_goes_down_near_a_wall():
    assert guard.wall_allowed(math.inf) == 1.0 and guard.wall_allowed(guard.WALL_SLOW_FROM_M) == 1.0
    assert guard.wall_allowed(guard.WALL_STOP_M) == 0.0 and guard.wall_allowed(-0.1) == 0.0
    assert guard.WALL_STOP_M < guard.STOP_M               # walls may come nearer than people


def test_the_rider_cannot_drive_into_a_wall_but_can_turn_away():
    sc = scene_with([])
    g = guard_for(sc)
    sc.lidar.getRangeImage = lambda: lidar_image(wall_across(CHAIR_RADIUS_M + 0.4))   # 0.4 m from the wall
    out = g.check(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert 0.0 < out[0] < 1.0 and out[2] == 1.0
    assert g.warning.startswith("OBSTACLE 0.4 m AHEAD") and "slowing" in g.warning
    assert sc.sup.ring.value in (None, guard.HIDDEN)      # the red ring is for people only
    sc.lidar.getRangeImage = lambda: lidar_image(wall_across(CHAIR_RADIUS_M + 0.05))  # 5 cm away
    out = g.check(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert out[0] == 0.0 and out[2] == 1.0 and "turn away" in g.warning
    sc.lidar.getRangeImage = lambda: lidar_image(wall_along(0.6))   # turned along the wall
    assert g.check(np.array([1.0, 0.0, 0.0], dtype=np.float32))[0] == 1.0 and g.warning == ""


def test_a_person_ahead_is_warned_about_before_the_wall_behind_them():
    sc = scene_with([person(1.0 + CHAIR_RADIUS_M + PERSON_DIAMETER_M / 2, 0.0)])     # 1 m gap
    g = guard_for(sc)
    sc.lidar.getRangeImage = lambda: lidar_image(wall_across(3.0))
    g.check(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert g.warning.startswith("PERSON")


def test_a_wall_hit_is_counted_once_and_apart_from_people():
    g = guard_for(scene_with([]))
    for wall, human in ((False, False), (True, False), (True, False), (False, False), (True, True)):
        g.check(np.zeros(3, dtype=np.float32), {"COLLISION": wall, "COLLISION_HUMAN": human})
    assert g.wall_hits == 1 and g.hits == 1


def test_the_rider_is_slowed_down_and_warned_but_can_turn():
    sc = scene_with([person(1.0 + CHAIR_RADIUS_M + PERSON_DIAMETER_M / 2 - 0.25, 0.0)])   # 0.75 m gap ahead
    g = guard_for(sc)
    out = g.check(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert 0.0 < out[0] < 1.0 and out[2] == 1.0          # slower forward, the turn as asked
    assert g.warning.startswith("PERSON 0.7 m AHEAD") and "slowing" in g.warning
    assert sc.sup.ring.value[:2] == [round(sc.crowd.everyone[0].x, 2), 0.0]
    sc.robot.x = 0.6                                     # 0.15 m gap: no forward at all
    assert g.check(np.array([1.0, 0.0, 0.0], dtype=np.float32))[0] == 0.0 and "turn away" in g.warning
    sc.robot.orientation = 2.0                           # turned away: full speed, no warning, no ring
    assert g.check(np.array([1.0, 0.0, 0.0], dtype=np.float32))[0] == 1.0
    assert g.warning == "" and sc.sup.ring.value == guard.HIDDEN


def test_without_the_brake_it_only_warns():
    sc = scene_with([person(1.5, 0.0)])
    g = guard_for(sc, brake=False)
    assert g.check(np.array([1.0, 0.0, 0.0], dtype=np.float32))[0] == 1.0
    assert g.warning.endswith("careful")


def test_a_hit_is_counted_once_however_long_it_lasts():
    g = guard_for(scene_with([]))
    for touching in (False, True, True, True, False, True):
        g.check(np.zeros(3, dtype=np.float32), {"COLLISION_HUMAN": touching})
    assert g.hits == 2


def test_text_that_changes_every_step_is_sent_less_often():
    from webots import run

    class Sup:
        def __init__(self):
            self.calls = 0

        def setLabel(self, *args):
            self.calls += 1

    sup = Sup()
    overlay = run.Overlay(sup)
    for k in range(10):                                  # ten steps in quick succession, a new speed each time
        overlay.lines([("MANUAL", 0), (f"{k / 10:.1f} m/s", 0)])
    assert sup.calls == 2 * 5                            # each line once: 4 white copies and the text
    assert overlay.pending                                # the newest speed waits ...
    time.sleep(run.TEXT_EVERY_S + 0.01)
    overlay.lines([("MANUAL", 0), ("0.9 m/s", 0)])
    assert sup.calls == 3 * 5 and not overlay.pending    # ... and is sent once it may be
    assert overlay.drawn[1][0] == "0.9 m/s"


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
