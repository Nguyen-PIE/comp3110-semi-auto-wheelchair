"""Checks for envs/behaviours.py: how fast people walk, phones, stopping, waiting, groups.

    python tests/test_behaviours.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import behaviours  # noqa: E402
from envs.make_env import make_env  # noqa: E402

STILL = np.zeros(3, dtype=np.float32)
NONE = {"phone_share": 0, "group_share": 0, "pause_share": 0, "dwell_chance": 0}


def run(overrides, steps=300, seed=2, scenario="s3_concourse_mid"):
    """Walk a scene with the chair parked; returns the scene and, per person, their speed each step."""
    env = make_env(scenario, overrides={"project": {"behaviour": overrides}})
    env.reset(seed=seed)
    scene = env.unwrapped
    speeds = {h.id: [] for h in scene.dynamic_humans}
    for _ in range(steps):
        env.step(STILL)
        for h in scene.dynamic_humans:
            speeds[h.id].append(h.speed)
    return scene, speeds


def test_walking_speeds_are_spread_out_and_within_limits():
    cfg = behaviours.settings()
    rng = random.Random(0)
    speeds = np.array([behaviours.draw_speed(rng, cfg) for _ in range(2000)])
    assert speeds.min() >= cfg["speed_min"] and speeds.max() <= cfg["speed_max"]
    assert abs(speeds.mean() - cfg["speed_mean"]) < 0.02
    assert speeds.std() > 0.15      # a mix of strollers and people in a hurry, not one pace


def test_speeds_never_exceed_the_environments_top_speed():
    cfg = behaviours.settings()
    scene, speeds = run({})
    assert max(max(s) for s in speeds.values()) <= scene.MAX_ADVANCE_HUMAN + 1e-6
    assert cfg["speed_max"] <= scene.MAX_ADVANCE_HUMAN


def test_waiting_on_arrival_delays_the_next_destination():
    person = type("P", (), {})()
    person._traits = behaviours.Traits(random.Random(1), dwell_chance=1.0, dwell_seconds=(5.0, 5.0))
    env = type("E", (), {"TIMESTEP": 0.1})()
    assert behaviours.arrived(person, False) is False
    assert behaviours.arrived(person, True) is False            # starts waiting
    for _ in range(49):
        assert not np.any(behaviours.hold(env, person, np.ones(2)))
        assert behaviours.arrived(person, True) is False
    behaviours.hold(env, person, np.ones(2))
    behaviours.hold(env, person, np.ones(2))
    assert behaviours.arrived(person, True) is True             # waited 5 s: free to choose a new one
    assert person._traits.dwell_left is None


def test_nobody_waits_when_waiting_is_switched_off():
    scene, _ = run(NONE, steps=5)
    person = scene.dynamic_humans[0]
    assert behaviours.arrived(person, True) is True


def test_people_stopping_at_random_stand_still_part_of_the_time():
    _, speeds = run({**NONE, "pause_share": 1.0, "pause_rate": [0.2, 0.2], "pause_seconds": [3.0, 3.0]})
    standing = np.mean([np.mean(np.array(s) < 0.05) for s in speeds.values()])
    assert 0.1 < standing < 0.9
    _, speeds = run(NONE)
    assert np.mean([np.mean(np.array(s) < 0.05) for s in speeds.values()]) < 0.05


def test_phone_walkers_are_slow_and_do_not_watch_for_the_chair():
    scene, speeds = run({**NONE, "phone_share": 1.0})
    for h in scene.dynamic_humans:
        assert behaviours.is_phone(h)
        assert not h.avoids_robot
        assert h._walk_speed <= behaviours.settings()["phone_speed"][1]


def test_group_members_stay_beside_their_leader():
    env = make_env("s3_concourse_mid", overrides={"project": {"behaviour": {**NONE, "group_share": 1.0}}})
    env.reset(seed=2)
    scene = env.unwrapped
    members = [h for h in scene.dynamic_humans if behaviours.traits_of(h).leader]
    assert members, "no groups were made"
    gaps = []
    for step in range(300):
        env.step(STILL)
        if step > 50:
            gaps += [math.dist((h.x, h.y), (behaviours.traits_of(h).leader.x, behaviours.traits_of(h).leader.y))
                     for h in members]
    assert np.mean(gaps) < 1.6, f"group members drifted apart (mean gap {np.mean(gaps):.2f} m)"
    assert np.mean(np.array(gaps) < 2.5) > 0.9


def test_group_members_start_clear_of_walls_and_each_other():
    """(The scene generator itself sometimes starts two unrelated people close together; only
    groups are checked here.)"""
    for seed in range(1, 11):
        env = make_env("s3_concourse_mid", overrides={"project": {"behaviour": {**NONE, "group_share": 1.0}}})
        env.reset(seed=seed)
        scene = env.unwrapped
        plan = scene._room_plan
        for h in scene.dynamic_humans:
            t = behaviours.traits_of(h)
            if t.leader is None:
                continue
            assert plan.clearance_at(h.x, h.y) > h.width / 2
            for other in [t.leader] + [m for m in t.leader._traits.members if m is not h]:
                assert math.dist((h.x, h.y), (other.x, other.y)) > 0.6, f"seed {seed}: group members overlap"


def test_same_seed_gives_the_same_crowd():
    first = [(h.x, h.y, h._walk_speed) for h in run({}, steps=1)[0].dynamic_humans]
    second = [(h.x, h.y, h._walk_speed) for h in run({}, steps=1)[0].dynamic_humans]
    assert first == second


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "-", exc)
    sys.exit(1 if failed else 0)
