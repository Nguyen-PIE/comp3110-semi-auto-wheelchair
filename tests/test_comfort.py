"""Checks for eval/comfort.py on made-up trips (no simulator needed).

    python tests/test_comfort.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.comfort import Trace, compute, count_weaves, stop_events  # noqa: E402

DT = 0.1


def straight_trip(speeds, person=None, seconds=None):
    """A chair driving along +x with the given speed profile. `person(t)` gives a person's (x, y)
    at step t, or None for nobody. The tracker's estimate is the true position and velocity."""
    tr = Trace(dt=DT, chair_radius=0.4, cruise=1.2)
    x = 0.0
    for t, s in enumerate(speeds):
        x += s * DT
        tr.x.append(x), tr.y.append(0.0), tr.heading.append(0.0), tr.v.append(s), tr.w.append(0.0)
        tr.safety_stop.append(False), tr.engaged.append(False)
        if person and person(t):
            px, py = person(t)
            nx, ny = person(t + 1) or (px, py)
            tr.people.append({1: (px, py, 0.36)})
            tr.tracks.append({1: (px, py, (nx - px) / DT, (ny - py) / DT, False)})
        else:
            tr.people.append({}), tr.tracks.append({})
    return tr


def profile(*parts):
    """[(speed, seconds), ...] -> speed per step."""
    return [s for s, sec in parts for _ in range(int(sec / DT))]


def test_smooth_drive_has_no_stops_or_hard_brakes():
    tr = straight_trip(profile((1.2, 20)))
    m = compute(tr, success=True)
    assert m["stops"] == 0 and m["hard_brakes"] == 0 and m["peak_decel_mps2"] == 0


def test_the_standstill_at_the_start_and_the_stop_at_the_goal_are_not_stops():
    tr = straight_trip(profile((0.0, 1), (1.2, 10), (0.0, 1)))
    assert compute(tr, success=True)["stops"] == 0
    assert compute(tr, success=False)["stops"] == 1   # stopping and never going again, without arriving


def test_stop_and_go_is_counted_and_a_pause_for_nothing_is_unneeded():
    # Driving, stopping for 2 s with nobody around, then driving on.
    tr = straight_trip(profile((1.2, 5), (0.0, 2), (1.2, 5)))
    m = compute(tr, success=True)
    assert m["stops"] == 1 and m["stop_go_cycles"] == 1
    assert m["stops_clear_of_people"] == 1 and m["unneeded_stops"] == 1


def test_a_stop_for_someone_really_crossing_is_needed():
    # A person walks across the chair's path 5 m ahead, timed to meet it; the chair stops for them.
    speeds = profile((1.2, 3), (0.0, 3), (1.2, 4))
    person = lambda t: (4.0, 3.0 - 1.0 * t * DT) if t < 90 else None   # noqa: E731  crosses y=0 at t=3 s
    m = compute(straight_trip(speeds, person), success=True)
    assert m["stops"] == 1 and m["stops_near_people"] == 1
    assert m["unneeded_stops"] == 0, m


def test_a_stop_for_someone_far_off_the_path_is_unneeded():
    speeds = profile((1.2, 3), (0.0, 3), (1.2, 4))
    person = lambda t: (4.0, 4.0) if t < 90 else None   # noqa: E731  standing 4 m to the side
    m = compute(straight_trip(speeds, person), success=True)
    assert m["stops"] == 1 and m["unneeded_stops"] == 1


def test_braking_instantly_is_a_hard_brake_and_slowing_gently_is_not():
    abrupt = straight_trip(profile((1.2, 5), (0.0, 2), (1.2, 3)))
    assert compute(abrupt, success=True)["hard_brakes"] == 1
    assert compute(abrupt, success=True)["peak_decel_mps2"] > 10   # 1.2 m/s in one 0.1 s step
    gentle = straight_trip([1.2] * 50 + [1.2 - 0.05 * i for i in range(1, 25)] + [0.0] * 30)
    assert compute(gentle, success=True)["hard_brakes"] == 0


def test_speed_near_people_is_judged_only_while_someone_is_near():
    person = lambda t: (6.0, 1.5) if t < 200 else None   # noqa: E731
    speeds = profile((1.2, 3), (0.5, 3), (1.2, 4))
    m = compute(straight_trip(speeds, person), success=True)
    assert 0.3 < m["min_speed_near_people"] < 0.5


def test_weaving():
    dt = DT
    # left, right, left within 1.5 s: one weave; the same turns spread over 6 s: none
    quick = [0.5] * 3 + [0] * 2 + [-0.5] * 3 + [0] * 2 + [0.5] * 3
    slow = [0.5] * 3 + [0] * 20 + [-0.5] * 3 + [0] * 20 + [0.5] * 3
    assert count_weaves(quick, dt) == 1 and count_weaves(slow, dt) == 0


def test_stop_events_ignore_flicker_between_the_two_thresholds():
    v = [0.0] * 5 + [1.0] * 5 + [0.2, 0.25, 0.15, 0.2] + [1.0] * 5   # never below 0.1: no stop
    assert stop_events(v, success=True) == []


def test_prediction_error_is_zero_for_a_person_walking_straight():
    person = lambda t: (2.0 + 1.0 * t * DT, 3.0)   # noqa: E731  walks along +x at 1 m/s
    tr = straight_trip(profile((1.0, 10)), person)
    m = compute(tr, success=False)
    assert m["pred_err_2s_m"] < 0.05


def test_prediction_error_is_large_for_a_person_who_turns():
    # walks along +x for 2 s, then turns and walks along +y: a straight-line guess is wrong
    def person(t):
        s = t * DT
        return (2.0 + s, 3.0) if s <= 2 else (4.0, 3.0 + (s - 2))
    tr = straight_trip(profile((1.0, 10)), person)
    assert compute(tr, success=False)["pred_err_2s_m"] > 0.3


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failed else 0)
