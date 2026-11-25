"""SiT track reading, replay and the track view's drawing and scoring (the parts that need no Webots).

The label files made here are synthetic, in the layout scripts/sit_tracks.py documents; they are
not SiT data, so these tests check the code, not that real SiT files read the same way.
"""

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sit_tracks  # noqa: E402
from webots import replay, trackview  # noqa: E402


def write_scene(folder: Path, speed: float = 1.2, frames: int = 60) -> None:
    """One pedestrian walking along x at `speed`, one car, one still person in the robot's frame."""
    folder.mkdir(parents=True)
    for f in range(frames):
        x = 2.0 + speed * f / 10
        (folder / f"{f:06d}.txt").write_text(
            f"Pedestrian 7 1.7 0.5 0.5 {x:.3f} 1.0 0.0 0.0\n"
            f"Car 9 1.5 4.0 1.8 10.0 5.0 0.0 0.0\n"
            f"Pedestrian 8 1.7 0.5 0.5 -3.0 2.0 0.0 0.0\n")


def test_read_labels_keeps_pedestrians_and_times(tmp_path):
    write_scene(tmp_path / "scene_a")
    table = sit_tracks.read_labels(tmp_path)
    assert set(table["id"]) == {"scene_a:7", "scene_a:8"}
    walker = table[table["id"] == "scene_a:7"].sort_values("t")
    assert math.isclose(walker["t"].iloc[-1], 5.9)
    speed = np.hypot(np.diff(walker["x"]), np.diff(walker["y"])) / np.diff(walker["t"])
    assert np.allclose(speed, 1.2, atol=1e-6)


def test_to_fixed_frame_removes_robot_motion(tmp_path):
    write_scene(tmp_path / "s", speed=0.0, frames=5)
    table = sit_tracks.read_labels(tmp_path)
    poses = pd.DataFrame({"scene": "s", "frame": range(5), "x": [i * 0.5 for i in range(5)], "y": 0.0, "yaw": 0.0})
    moved = sit_tracks.to_fixed_frame(table, poses)
    still = moved[moved["id"] == "s:8"].sort_values("t")
    assert np.allclose(np.diff(still["x"]), 0.5)           # still in the robot's frame, so moving in the world


def test_summary_reports_speed(tmp_path):
    write_scene(tmp_path / "s")
    text = sit_tracks.summary(sit_tracks.read_labels(tmp_path))
    assert "1.20 m/s" in text or "median walking speed" in text


def make_plan(size_m: float = 20.0):
    res = 0.05
    n = int(size_m / res)
    cells = np.zeros((n, n), np.uint8)
    cells[:10, :] = cells[-10:, :] = cells[:, :10] = cells[:, -10:] = 1
    from envs.floorplan import build_plan
    return build_plan("test", res, cells == 1, cells == 0, (0.0, 0.0))


def test_replay_fits_inside_the_plan_and_keeps_speed():
    plan = make_plan()
    t = np.arange(0, 12, 0.1)
    track = replay.Track("a", t, 1.3 * t, 0.2 * np.sin(t))
    placed = replay.fit(plan, track, plan.spawn_cells, random.Random(1), [])
    assert placed is not None
    xs, ys, count = placed
    assert count > 40
    assert all(plan.clearance_at(float(x), float(y)) >= replay.CLEARANCE_M for x, y in zip(xs, ys))
    step = np.hypot(np.diff(xs), np.diff(ys)) / 0.1
    assert np.allclose(step, np.hypot(np.diff(track.x[:count]), np.diff(track.y[:count])) / 0.1)


def test_replayed_position_velocity_and_end():
    t = np.arange(0, 6, 0.1)
    track = replay.Track("a", t, t, np.zeros_like(t))
    r = replay.Replayed(track, 2 * t, np.zeros_like(t), start=0.0)       # moved and scaled: 2 m/s
    person = SimpleNamespace(x=0.0, y=0.0)
    v = r.velocity(person, 0.0, 0.1)
    assert np.allclose(v, [2.0, 0.0])
    assert r.position(7.0) is None and r.velocity(person, 6.5, 0.1) is None
    assert len(r.future(1.0, 3.0, 0.5)) == 6
    assert len(r.future(4.0, 3.0, 0.5)) < 6                               # the track runs out


def test_load_drops_short_tracks(tmp_path):
    rows = [("long", i * 0.1, i * 0.1, 0.0) for i in range(80)] + [("short", i * 0.1, 0.0, 0.0) for i in range(10)]
    pd.DataFrame(rows, columns=["id", "t", "x", "y"]).to_csv(tmp_path / "t.csv", index=False)
    tracks = replay.load(tmp_path / "t.csv")
    assert [t.id for t in tracks] == ["long"] and tracks[0].t[0] == 0.0


def test_ribbon_is_a_strip_that_faces_up():
    corners = trackview.ribbon([(0, 0), (1, 0), (2, 0)], half_width=0.1)
    assert len(corners) == 6
    left, right = corners[0], corners[1]
    assert left[1] > right[1]                                              # left of +x is +y
    assert trackview.ribbon([(0, 0)]) is None
    assert trackview.ribbon([(1, 1), (1, 1), (1, 1)]) is None             # went nowhere: nothing to draw
    nudged = trackview.ribbon([(0, 0), (1, 0), (1, 0), (2, 0)])            # a pause: moved on, never flat
    xs = [c[0] for c in nudged[::2]]
    assert all(b - a >= trackview.MIN_SEGMENT_M - 1e-9 for a, b in zip(xs, xs[1:]))
    assert trackview.faces(3) == "0 1 3 2 -1 2 3 5 4 -1"


def test_behaviour_names():
    from envs import behaviours
    base = dict(kind="person", speed=1.0, type="dynamic", player=None, elderly=False, _traits=None)
    assert trackview.behaviour_of(SimpleNamespace(**base)) == "walking"
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "kind": "pet"})) == "other"
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "elderly": True})) == "elderly"
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "speed": 0.0})) == "stopped"
    traits = behaviours.Traits(random.Random(0), phone=True)
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "_traits": traits})) == "phone"
    traits = behaviours.Traits(random.Random(0), leader=object())
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "_traits": traits})) == "group"
    traits = behaviours.Traits(random.Random(0), pause_left=2.0)
    assert trackview.behaviour_of(SimpleNamespace(**{**base, "_traits": traits})) == "stopped"


def test_scoring_ade_fde():
    """A person who walks straight is predicted exactly; one who turns is not."""
    view = trackview.TrackView.__new__(trackview.TrackView)
    view.scene = SimpleNamespace(TIMESTEP=0.1, ticks=0)
    view.reset()
    person = SimpleNamespace(id=1, x=0.0, y=0.0, kind="person", speed=1.0, type="dynamic", player=None,
                             elderly=False, _traits=None)
    track = SimpleNamespace(id=1, x=0.0, y=0.0, vx=1.0, vy=0.0)
    for tick in range(0, 40):
        view.scene.ticks = tick
        person.x, person.y = tick * 0.1, 0.0                              # walking exactly as predicted
        view._score(tick * 0.1, [track] if tick == 0 else [], {1: person})
    ade, fde, n = view.scores()
    assert n == 1 and ade < 1e-6 and fde < 1e-6

    view.reset()
    for tick in range(0, 40):
        view.scene.ticks = tick
        person.x, person.y = tick * 0.1, 0.5 * (tick * 0.1)                # veers off at 0.5 m per metre
        view._score(tick * 0.1, [track] if tick == 0 else [], {1: person})
    ade, fde, n = view.scores()
    assert n == 1 and fde > ade > 0.1
    assert abs(fde - 0.5 * 3.0) < 1e-6                                     # 1.5 m sideways at 3 s


def test_replayed_person_waits_for_the_chair_and_never_jumps():
    from types import SimpleNamespace
    from webots import people
    t = np.arange(0, 10, 0.1)
    track = replay.Track("a", t, t, np.zeros_like(t))           # walks along x at 1 m/s
    person = SimpleNamespace(x=0.0, y=0.0, width=0.5, _replay=replay.Replayed(track, t.copy(), np.zeros_like(t), 0.0))
    chair = SimpleNamespace(x=3.0, y=0.0, radius=0.5)
    crowd = people.Crowd.__new__(people.Crowd)
    crowd.scene = SimpleNamespace(robot=chair)
    crowd.walking, crowd.standing = [person], []
    dt, biggest = 0.1, 0.0
    for tick in range(300):
        if tick == 150:
            chair.x = 20.0                                       # the chair drives off
        v = person._replay.velocity(person, tick * dt, dt)
        if v is None:
            break
        if crowd._blocked(person, v, dt):
            person._replay.start += dt
            continue
        speed = float(np.hypot(*v))
        if speed > people.REPLAY_MAX_MPS:
            person._replay.start += dt * (1 - people.REPLAY_MAX_MPS / speed)
            v = v * people.REPLAY_MAX_MPS / speed
        biggest = max(biggest, float(np.hypot(*v)))
        person.x, person.y = person.x + v[0] * dt, person.y + v[1] * dt
        assert math.dist((person.x, person.y), (chair.x, chair.y)) > chair.radius + 0.25 - 1e-6
    assert biggest <= people.REPLAY_MAX_MPS + 1e-6


def test_reads_sits_published_trajectory_txt(tmp_path):
    """SiT's GitHub trajectory_txt layout: one file per scene, "frame class id x y yaw"."""
    scene = tmp_path / "Corridor" / "Corridor_1"
    scene.mkdir(parents=True)
    lines = []
    for f in range(30):
        lines += [f"{f} husky 0 {0.05 * f} 0.0 0.0", f"{f} Pedestrian 3 {1.0 + 0.12 * f} 2.0 0.0",
                  f"{f} Pedestrain_sitting 10004 5.0 5.0 0.0"]
    (scene / "trajectory.txt").write_text("\n".join(lines) + "\n")
    table = sit_tracks.read_labels(tmp_path)
    assert set(table["id"]) == {"Corridor_1:ID:3"}
    assert math.isclose(table["t"].max(), 2.9)


def test_groups_are_only_counted_within_one_scene():
    """Two scenes both start at t = 0: people in different scenes at the same x, y are not a group."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import fit_behaviour
    t = np.arange(0, 6, 0.1)
    rows = [(f"{scene}:ID:1", tt, 1.2 * tt, 0.0) for scene in ("A_1", "B_1") for tt in t]
    frame = pd.DataFrame(rows, columns=["id", "t", "x", "y"])
    assert fit_behaviour.share_in_groups(frame) == 0.0
    frame.loc[frame["id"] == "B_1:ID:1", "id"] = "A_1:ID:2"      # same scene now: walking together
    assert fit_behaviour.share_in_groups(frame) == 1.0
