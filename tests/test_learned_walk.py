"""Checks for the learned walkers (webots/learned_walk.py, --walkers learned). No Webots needed:
the stub scene from test_streaming.py stands in for it. Skipped when Trajectron++ or the trained
model is missing (bash webots/walk_model/get_trajectron.sh; the model is in models/walk_trajectron/).

    python tests/test_learned_walk.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from webots import learned_walk as LW  # noqa: E402

MISSING = LW.available()
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = LW.LearnedWalkers()
    return _MODEL


def _trip(seed, seconds=20.0, people=12, chair_near=False):
    from test_streaming import Scene
    scene = Scene(seed=seed, scale=1.0)
    scene.start(people)
    if not chair_near:
        scene.robot.x = scene.robot.y = 1e3              # chair out of the way
    scene.crowd.learned = _model()
    _model().reset(seed)
    for _ in range(int(seconds / scene.TIMESTEP)):
        scene.crowd.step(scene.TIMESTEP)
    return scene


def _skip():
    if MISSING:
        print("skip  (", MISSING, ")")
        try:
            import pytest
            pytest.skip(MISSING)
        except ImportError:
            pass
        return True
    return False


def test_the_same_seed_replays_the_same_walk():
    if _skip():
        return
    a = [(round(p.x, 6), round(p.y, 6)) for p in _trip(7).crowd.everyone]
    b = [(round(p.x, 6), round(p.y, 6)) for p in _trip(7).crowd.everyone]
    assert a == b, "two runs of seed 7 put people in different places"


def test_people_keep_out_of_walls_and_off_the_chair():
    if _skip():
        return
    scene = _trip(3, seconds=30.0, chair_near=True)
    chair = scene.robot
    for p in scene.crowd.walking:
        assert scene.plan.clearance_at(p.x, p.y) > 0.0, f"person {p.id} is inside a wall"
        assert math.dist((p.x, p.y), (chair.x, chair.y)) >= chair.radius + p.width / 2 - 1e-6


def test_people_get_where_they_are_going():
    if _skip():
        return
    from test_streaming import Scene
    scene = Scene(seed=11, scale=1.0)
    scene.start(12)
    scene.robot.x = scene.robot.y = 1e3
    scene.crowd.learned = _model()
    _model().reset(11)
    arrived = 0
    for _ in range(600):                              # 60 s
        before = {p.id: scene.goals.get(p.id) for p in scene.crowd.walking}
        scene.crowd.step(scene.TIMESTEP)
        arrived += sum(1 for p in scene.crowd.walking
                       if before.get(p.id) is not None and scene.goals.get(p.id) is not before[p.id])
    assert arrived >= 6, f"only {arrived} arrivals by 12 people in 60 s"


def test_the_model_steers_ordinary_people_only():
    if _skip():
        return
    scene = _trip(5, seconds=2.0)
    model = _model()
    people = [p for p in scene.crowd.walking if p.kind == "person"]
    assert people and all(model.velocity(p) is not None for p in people)
    pet = people[0]
    pet.kind = "pet"
    model.update(scene.crowd, 0.4)
    assert model.velocity(pet) is None


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
