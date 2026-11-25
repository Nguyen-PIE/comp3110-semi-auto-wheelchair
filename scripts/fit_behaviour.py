"""Estimate the crowd settings in envs/configs/base.yaml (project.behaviour) from real pedestrian data.

Reads a table of pedestrian positions over time and prints the walking speed, stopping and group
values it implies, as a block to paste into base.yaml. Two input layouts:

    --format eth      the ETH/UCY "obsmat.txt" of OpenTraj (frame, id, x, z, y, vx, vz, vy; whitespace)
    --format table    any whitespace or comma separated file with --columns id,t,x,y (default
                      order: id, time in seconds, x, y in metres); --time-scale converts frames
                      to seconds (for example 0.04 for 25 frames per second)

Examples:
    python scripts/fit_behaviour.py datasets/ETH/seq_hotel/obsmat.txt --format eth
    python scripts/fit_behaviour.py thor_day1.csv --format table --columns id,t,x,y

What it can and cannot tell you: speed, stops and groups come from the positions. Whether someone is
on the phone does not, so phone_share and the phone settings stay as they are. Outdoor datasets
(ETH, UCY) walk faster than people indoors, so prefer indoor data (THOR, ATC, Grand Central) for a
hospital or office; use ETH only as a rough check.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MOVING_MPS = 0.2        # slower than this counts as standing
MIN_TRACK_S = 4.0       # ignore tracks shorter than this
MIN_STOP_S = 1.0        # a stop is at least this long
GROUP_GAP_M = 1.6       # walking together: this close ...
GROUP_SPEED_DIFF = 0.3  # ... with speeds this alike (m/s) ...
GROUP_MIN_S = 3.0       # ... for this long


def load(path: str, fmt: str, columns: str, fps: float, time_scale: float | None) -> pd.DataFrame:
    if fmt == "eth":
        raw = pd.read_csv(path, sep=r"\s+", header=None)
        return pd.DataFrame({"id": raw[1].astype(int), "t": raw[0] / (fps or 25.0), "x": raw[2], "y": raw[4]})
    names = [c.strip() for c in columns.split(",")]
    raw = pd.read_csv(path, sep=None, engine="python")
    frame = raw[names].copy() if all(n in raw.columns for n in names) else \
        pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :4].set_axis(names, axis=1)
    frame = frame.rename(columns=dict(zip(names, ["id", "t", "x", "y"])))
    if time_scale:
        frame["t"] = frame["t"] * time_scale
    return frame


def tracks(frame: pd.DataFrame):
    """Yield (id, t, x, y, speed) arrays for each long enough track."""
    for pid, g in frame.sort_values("t").groupby("id"):
        t, x, y = g["t"].to_numpy(float), g["x"].to_numpy(float), g["y"].to_numpy(float)
        if len(t) < 3 or t[-1] - t[0] < MIN_TRACK_S:
            continue
        dt = np.diff(t)
        ok = dt > 0
        speed = np.hypot(np.diff(x), np.diff(y))[ok] / dt[ok]
        yield pid, t, x, y, np.concatenate([[speed[0]], speed]) if len(speed) else np.zeros(len(t))


def stops(t, speed) -> list[float]:
    """Lengths (s) of the stretches where the person stood still."""
    out, start = [], None
    for k, slow in enumerate(speed < MOVING_MPS):
        if slow and start is None:
            start = t[k]
        elif not slow and start is not None:
            out.append(t[k] - start)
            start = None
    return [d for d in out if d >= MIN_STOP_S]


def scene_of(pid) -> str:
    """The recording a track comes from: scripts/sit_tracks.py names them "<scene>:ID:<n>". Every SiT
    scene starts at t = 0 in its own coordinates, so people are only compared within one scene."""
    return str(pid).rsplit(":", 1)[0] if ":" in str(pid) else ""


def share_in_groups(frame: pd.DataFrame) -> float:
    """Share of people who spend at least GROUP_MIN_S close beside someone moving the same way."""
    times = {}
    for pid, t, x, y, speed in tracks(frame):
        scene = scene_of(pid)
        for k in range(len(t)):
            times.setdefault((scene, round(t[k], 3)), []).append((pid, x[k], y[k], speed[k]))
    together: dict = {}
    steps = sorted({t for _, t in times})
    dt = float(np.median(np.diff(steps))) if len(steps) > 1 else 0.4
    for rows in times.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                (i, xi, yi, si), (j, xj, yj, sj) = rows[a], rows[b]
                if si > MOVING_MPS and sj > MOVING_MPS and math.hypot(xi - xj, yi - yj) < GROUP_GAP_M \
                        and abs(si - sj) < GROUP_SPEED_DIFF:
                    together[(i, j)] = together.get((i, j), 0.0) + dt
    in_group = {p for pair, s in together.items() if s >= GROUP_MIN_S for p in pair}
    people = {pid for pid, *_ in tracks(frame)}
    return len(in_group & people) / max(1, len(people))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--format", choices=["eth", "table"], default="table")
    ap.add_argument("--columns", default="id,t,x,y")
    ap.add_argument("--fps", type=float, default=None, help="eth: video frames per second of the frame column (default 25)")
    ap.add_argument("--time-scale", type=float, default=None, help="table: multiply the time column by this to get seconds")
    args = ap.parse_args()

    frame = load(args.path, args.format, args.columns, args.fps, args.time_scale)
    walking, stop_rows, track_stops, seconds = [], [], 0, 0.0
    n_tracks = 0
    for pid, t, x, y, speed in tracks(frame):
        n_tracks += 1
        moving = speed[speed >= MOVING_MPS]
        if len(moving):
            walking.append(float(np.mean(moving)))
        s = stops(t, speed)
        track_stops += bool(s)
        stop_rows += s
        seconds += t[-1] - t[0]
    if not walking:
        sys.exit("No usable tracks found.")
    walking = np.array(walking)
    rate = len(stop_rows) / seconds if seconds else 0.0
    print(f"{n_tracks} tracks, {seconds / 60:.1f} track-minutes")
    print(f"walking speed per person: mean {walking.mean():.2f} m/s, sd {walking.std():.2f}, "
          f"5th to 95th percentile {np.percentile(walking, 5):.2f} to {np.percentile(walking, 95):.2f}")
    print(f"people who stopped at least once: {track_stops / n_tracks:.0%}; "
          f"stops per person-second {rate:.3f}; typical stop {np.median(stop_rows) if stop_rows else 0:.1f} s "
          f"(90th percentile {np.percentile(stop_rows, 90) if stop_rows else 0:.1f} s)")
    grouped = share_in_groups(frame)
    print(f"people walking beside someone: {grouped:.0%}\n")
    lo, hi = (np.percentile(stop_rows, 25), np.percentile(stop_rows, 90)) if stop_rows else (2.0, 8.0)
    print("Suggested values for project.behaviour in envs/configs/base.yaml:")
    print(f"  speed_mean: {walking.mean():.2f}\n  speed_sd: {walking.std():.2f}\n"
          f"  speed_min: {max(0.4, np.percentile(walking, 1)):.2f}\n  speed_max: {np.percentile(walking, 99):.2f}\n"
          f"  group_share: {grouped:.2f}\n  pause_share: {track_stops / n_tracks:.2f}\n"
          f"  pause_rate: [{rate * 0.5:.3f}, {rate * 1.5:.3f}]\n  pause_seconds: [{lo:.1f}, {hi:.1f}]")
    print("\n(max_advance_human in the human: section must stay at or above speed_max.)")


if __name__ == "__main__":
    main()
