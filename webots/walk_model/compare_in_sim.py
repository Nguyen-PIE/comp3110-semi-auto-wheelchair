"""Rules or learned walkers: how do people in the simulator walk, next to real people?

Runs the crowd code (webots/people.py, with envs/pedestrians.py on top) without Webots, on a floor
plan, with the chair parked, and records how people walk: their speed, how sharply they turn, how
close they come to each other, how often they get where they are going, and whether anyone ends up
inside a wall. The same numbers are worked out for the real people in the ETH/UCY recordings.

    python webots/walk_model/compare_in_sim.py [--seconds 120] [--trips 3] [--people 20]
"""

from __future__ import annotations

import argparse
import glob
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from envs import pedestrians  # noqa: E402
from webots import learned_walk as LW  # noqa: E402

RAW = ROOT / "third_party/Trajectron-plus-plus/experiments/pedestrians/raw"
MOVING = 0.2    # m/s: slower than this counts as standing, and is left out of speed and turning


def stats(tracks: list[np.ndarray], dt: float, gaps: list[float]) -> dict:
    """tracks: positions every dt seconds, one array per person."""
    speeds, turns = [], []
    for xy in tracks:
        if len(xy) < 3:
            continue
        v = np.diff(xy, axis=0) / dt
        s = np.linalg.norm(v, axis=1)
        speeds += list(s[s > MOVING])
        heading = np.arctan2(v[:, 1], v[:, 0])
        ok = (s[1:] > MOVING) & (s[:-1] > MOVING)
        turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi) / dt
        turns += list(np.degrees(turn[ok]))
    speeds, turns, gaps = np.array(speeds), np.array(turns), np.array(gaps)
    return {"speed m/s (mean)": speeds.mean(), "speed m/s (spread)": speeds.std(),
            "turning deg/s (median)": np.median(turns),
            "closest person m (median)": np.median(gaps) if len(gaps) else float("nan"),
            "closer than 0.5 m (share)": (gaps < 0.5).mean() if len(gaps) else float("nan")}


def real_people() -> dict:
    tracks, gaps = [], []
    for path in glob.glob(str(RAW / "*" / "test" / "*.txt")):
        d = np.loadtxt(path)
        frames = {}
        for f, i, x, y in d:
            frames.setdefault(int(f), []).append((int(i), x, y))
        by_id = {}
        for f in sorted(frames):
            pts = frames[f]
            xy = np.array([(x, y) for _, x, y in pts])
            for k, (i, x, y) in enumerate(pts):
                by_id.setdefault(i, []).append((x, y))
                if len(pts) > 1:
                    gaps.append(float(np.delete(np.linalg.norm(xy - xy[k], axis=1), k).min()))
        tracks += [np.array(t) for t in by_id.values()]
    return stats(tracks, 0.4, gaps)


def simulated(walkers: str, seconds: float, trips: int, people: int) -> dict:
    from test_streaming import Scene, PLAN
    model = LW.LearnedWalkers() if walkers == "learned" else None
    tracks, gaps, arrivals, in_walls, steps, cpu = [], [], 0, 0, 0, 0.0
    every = round(0.4 / Scene.TIMESTEP)   # sample like the recordings, every 0.4 s
    for trip in range(trips):
        scene = Scene(seed=1000 + trip, scale=1.0)
        scene.start(people)
        scene.robot.x, scene.robot.y = 1e3, 1e3          # chair out of the way: just people
        crowd = scene.crowd
        crowd.learned = model
        if model:
            model.reset(1000 + trip)
        trace: dict[int, list] = {}
        goals = {p.id: (p.goal_x, p.goal_y) for p in crowd.walking}
        for tick in range(int(seconds / Scene.TIMESTEP)):
            before = {p.id: scene.goals.get(p.id) for p in crowd.walking}
            t0 = time.perf_counter()
            crowd.step(Scene.TIMESTEP)
            cpu += time.perf_counter() - t0
            steps += 1
            arrivals += sum(1 for p in crowd.walking if scene.goals.get(p.id) is not before.get(p.id)
                            and before.get(p.id) is not None)
            for p in crowd.walking:
                if PLAN.clearance_at(p.x, p.y) < 0.05:
                    in_walls += 1
            if tick % every == 0:
                xy = np.array([(p.x, p.y) for p in crowd.walking])
                for k, p in enumerate(crowd.walking):
                    trace.setdefault(p.id, []).append((p.x, p.y))
                    if len(xy) > 1:
                        gaps.append(float(np.delete(np.linalg.norm(xy - xy[k], axis=1), k).min()))
        tracks += [np.array(t) for t in trace.values()]
    out = stats(tracks, 0.4, gaps)
    out["destinations reached per person-minute"] = arrivals / (trips * people * seconds / 60)
    out["steps inside a wall (share)"] = in_walls / max(1, steps * people)
    out["ms per 0.1 s step"] = 1000 * cpu / steps
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--trips", type=int, default=3)
    p.add_argument("--people", type=int, default=20)
    args = p.parse_args(argv)
    columns = {"real people (ETH/UCY)": real_people(),
               "rules": simulated("rules", args.seconds, args.trips, args.people),
               "learned": simulated("learned", args.seconds, args.trips, args.people)}
    rows = list(dict.fromkeys(k for c in columns.values() for k in c))
    print(f"{'':42}" + "".join(f"{c:>24}" for c in columns))
    for r in rows:
        print(f"{r:42}" + "".join(f"{columns[c].get(r, float('nan')):24.2f}" for c in columns))


if __name__ == "__main__":
    main()
