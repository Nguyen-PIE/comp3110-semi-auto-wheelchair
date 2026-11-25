"""Does the chair stop short of a wall when the rider drives it at one? For each trip the rider
takes over (O), turns until a wall is 1 to 3 m straight ahead, then holds the up arrow for 6 s, as a
rider who never lets go would. With the wall brake (webots/guard.py) the chair should stop short;
with --no-brake it is driven on into the wall. No window: the guard is put on the rider's commands
here as run.py does in the window.

    python webots/override_check.py                  # 10 trips in fp_hospital, wall brake on
    python webots/override_check.py --no-brake       # the same trips without it
    python webots/override_check.py --world fp_office --trips 20

Each trip prints the smallest gap the chair's lidar saw between the chair's front and a wall
straight ahead while it drove (webots/guard.py's wall_ahead), how close the chair's centre came to a
wall on the floor plan, and whether the trip counted a wall hit. The centre is 0.40 m from a wall
(CHAIR_RADIUS_M) when the chair touches it. The trip's own rule counts a hit only below 0.40 m, which
a chair stopped by a solid Webots wall never reaches, so touching is counted here at 0.40 m (the floor
plan's 5 cm grid). Some trips turn into open space and never meet a wall: the summary leaves them out.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AIM_FROM_M, AIM_TO_M = 1.0, 3.0      # a wall this far straight ahead is the one to drive at
DRIVE_S = 6.0                        # forward held this long
REACHED_M = 0.3                      # a trip "reached a wall" when the gap ahead fell below this
UP, LEFT, O, Q = 315, 314, ord("O"), ord("Q")


def one_trip(world: str, seed: int, brake: bool, port: int) -> None:
    """Runs in its own process: one Webots, one trip, one RESULT line."""
    from webots import bridge, guard as guard_mod, run
    from webots.bridge import CHAIR_RADIUS_M
    s = {"n": 0, "phase": "start", "t": 0.0, "nearest": math.inf, "gap": math.inf, "guard": None}
    manual0, keys0 = run.manual_command, bridge.WebotsScene.keys

    def manual_command(keys, keyboard):
        command = manual0(keys, keyboard)
        return s["guard"].check(command, None) if brake and s["guard"] is not None else command

    def keys(scene):
        held = set(keys0(scene))
        s["n"] += 1
        now, r = scene.sup.getTime(), scene.robot
        if s["guard"] is None:
            s["guard"] = guard_mod.Guard(scene, brake=True)
        if s["phase"] == "start" and s["n"] >= 10:
            held.add(O)
            s["phase"], s["t"] = "aim", now
        elif s["phase"] == "aim":
            ahead = scene.lidar_ranges()[0]
            if AIM_FROM_M < ahead < AIM_TO_M or now - s["t"] > 15:
                s["phase"], s["t"] = "drive", now
            else:
                held.add(LEFT)
        elif s["phase"] == "drive":
            held.add(UP)
            s["nearest"] = min(s["nearest"], scene.plan.clearance_at(r.x, r.y))
            s["gap"] = min(s["gap"], s["guard"].wall_gap())
            if now - s["t"] > DRIVE_S:
                print(f"RESULT seed={seed} nearest={s['nearest']:.3f} hit={s['nearest'] < CHAIR_RADIUS_M} "
                      f"min_gap={s['gap']:.3f}", flush=True)
                s["phase"] = "end"
        if s["phase"] == "end":
            held.add(Q)
        return held

    run.manual_command, bridge.WebotsScene.keys = manual_command, keys
    run.main([world, "--seed", str(seed), "--port", str(port), "--fast", "--headless", "--crowd-scale", "0"]
             + ([] if brake else ["--no-brake"]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default="fp_hospital")
    p.add_argument("--trips", type=int, default=10)
    p.add_argument("--seed", type=int, default=10000, help="first seed (validation seeds, as the test set)")
    p.add_argument("--no-brake", action="store_true", help="drive at the wall without the wall brake")
    p.add_argument("--port", type=int, default=1350)
    p.add_argument("--one", type=int, help=argparse.SUPPRESS)       # (a single trip, run by main)
    args = p.parse_args(argv)
    if args.one is not None:
        one_trip(args.world, args.one, not args.no_brake, args.port)
        return
    from webots.bridge import CHAIR_RADIUS_M
    rows = []
    for seed in range(args.seed, args.seed + args.trips):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--world", args.world, "--one", str(seed),
               "--port", str(args.port)] + (["--no-brake"] if args.no_brake else [])
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
        m = re.search(r"RESULT seed=(\d+) nearest=([\d.]+|inf) hit=(\w+) min_gap=([\d.\-]+|inf)", out)
        if not m:
            print(f"seed {seed}: no result", flush=True)
            continue
        nearest, hit, gap = float(m.group(2)), m.group(3) == "True", float(m.group(4))
        rows.append((nearest, hit, gap))
        touch = nearest < CHAIR_RADIUS_M + 0.01
        print(f"seed {seed}: smallest gap ahead {gap:.2f} m, centre {nearest:.2f} m from a wall"
              f"{' (touching)' if touch else ''}, {'counted as a wall hit' if hit else 'no wall hit counted'}",
              flush=True)
    if rows:
        met = [(n, h, g) for n, h, g in rows if g < REACHED_M]
        hits = sum(h for _, h, _ in rows)
        print(f"\n{len(rows)} trips in {args.world}, wall brake {'off' if args.no_brake else 'on'}: "
              f"{len(met)} reached a wall ({len(rows) - len(met)} turned into open space)")
        if met:
            gaps = sorted(g for _, _, g in met)
            touched = sum(n < CHAIR_RADIUS_M + 0.01 for n, _, _ in met)
            print(f"  touched it (centre at {CHAIR_RADIUS_M:.2f} m): {touched} of {len(met)}, centre closest "
                  f"{min(n for n, _, _ in met):.2f} m, smallest gap ahead {gaps[0]:.2f} to {gaps[-1]:.2f} m "
                  f"(median {gaps[len(gaps) // 2]:.2f} m)")
        print(f"  counted as a wall hit by the trip's rule: {hits}")


if __name__ == "__main__":
    main()
