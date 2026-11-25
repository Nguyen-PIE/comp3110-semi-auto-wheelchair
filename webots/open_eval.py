"""Finding a room with the floor plan and without it: the same starts and destinations, both ways.

    python webots/open_eval.py                         # the hospital, 10 trips each way
    python webots/open_eval.py --trips 20 --read tags  # names from the sign tags instead of OCR
    python webots/open_eval.py --headless

Each trip starts where run.py's trip with that seed starts (people around, with run.py's settings)
and goes to a destination the building's directory lists (a department or a public place, at least
10 m away as the crow flies), picked at random:
    with plan   the chair is given the floor plan and the destination's spot, and drives the route
                (run.py's way: what a real chair with a map of the building would do)
    signs       the open world's way: no floor plan, no list of rooms. The chair first finds and reads
                a directory, then the destination is picked from it, as a rider would; the chair
                explores, follows the signs for its wing and department, and stops when it reads its
                sign (a department: its own sign, or a room whose name says it is in it).
Both are driven by the learned assistant (RL), with the chair's usual 4 m lidar. Per way it prints
how often the chair got there within the time limit, how long it took (for "signs" including
finding the directory), how far it drove, collisions, and for "signs" how long the directory took,
how much of the floor it had explored and how many places it had read, and what each collision was
(how near a wall and the nearest person). Every trip is saved as it ends, to
data/signs/<world>_open.csv or --out (so long runs can be split: --trips 5 --seed 10005).
For "signs" it also says how long it took from the directory to the place, how often and how long
it stopped to look closer at a sign, and how long it was inside rooms. --given-directory starts the
search with the directory known, --ocr-s 0.2 reads signs as late as run.py's reading thread does,
and --trace saves what it did every second (data/signs/trace/).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import launch  # noqa: E402

ARRIVED_M = 0.8


def listed(layout) -> list[str]:
    """What the building's directories list (the true answer)."""
    return sorted({name for f in layout.faces if f.kind == "directory" for name, _ in f.rows})


def at(layout, name) -> list:
    """Where the chair may stop for `name`: its own stop, or (a department) any of its places'."""
    groups = {f.place: f.group for f in layout.faces if f.place}
    return [layout.stops[p] for p in layout.stops if p == name or groups.get(p) == name]


def trips(layout, scene, n, seed, rng):
    """(seed, destination) pairs: the start is the trip's, the destination 10 m or more away."""
    names = listed(layout)
    out = []
    s = seed
    while len(out) < n:
        scene.reset(seed=s)
        here = (scene.robot.x, scene.robot.y)
        far = [x for x in names if x in layout.stops and math.dist(here, layout.stops[x]) >= 10.0]
        if far:
            out.append((s, rng.choice(far)))
        s += 1
    return out


def with_plan(scene, driver, layout, seed, name, limit):
    scene.reset(seed=seed)
    driver.reset()
    scene.time_limit = math.inf
    stop = layout.stops[name]                        # its own sign's stop (a department: its entrance sign)
    scene.set_goal(*stop)
    t0, path, last, hits = scene.sup.getTime(), 0.0, (scene.robot.x, scene.robot.y), 0
    while scene.sup.getTime() - t0 < limit:
        _, _, _, _, info = scene.step(driver.act(None))
        here = (scene.robot.x, scene.robot.y)
        path += math.dist(last, here)
        last = here
        hits += bool(info["COLLISION"])
        if math.dist(here, stop) < ARRIVED_M:
            return dict(arrived=1, time=scene.sup.getTime() - t0, path=path, collisions=hits)
    return dict(arrived=0, time=limit, path=path, collisions=hits)


def with_signs(session, seed, name, limit, layout, trace: Path | None = None):
    sc = session.scene
    session.reset(seed)                              # looking for a directory
    t0 = sc.sup.getTime()
    directory_s, path, hits = None, 0.0, 0
    seen_hits = 0
    steps = []
    looks = {"doors": 0, "other": 0, "s": 0.0, "rooms": 0.0}   # after the directory: stops to look closer at a
    looking = None                                   # sign, and time in rooms (off the corridors, from the true plan)
    geo = rooms_of(sc.plan)
    while sc.sup.getTime() - t0 < limit:
        session.step()
        if directory_s is not None and session.looking_at is not None:
            if session.looking_at is not looking:      # a new look
                s = session.reader.seen.get(session.looking_at["tag"])
                looks["doors" if s is not None and s.kind in ("plate", "blade") else "other"] += 1
            looks["s"] += 0.1
        if directory_s is not None and in_a_room(geo, sc.robot.x, sc.robot.y):
            looks["rooms"] += 0.1
        looking = session.looking_at
        if trace is not None and session.ticks % 10 == 0:   # once a second: what it was doing, for --trace
            L, r = session.looking_at, sc.robot
            s = session.reader.seen.get(L["tag"]) if L is not None else None
            steps.append(dict(t=round(sc.sup.getTime() - t0, 1), x=round(r.x, 2), y=round(r.y, 2),
                              speed=round(r.vel_x, 2), mode=session.mode,
                              looking=f"{s.kind if s is not None else '?'}:{L['phase']}" if L is not None else "",
                              room=int(in_a_room(geo, r.x, r.y)), backing=int(session.backing is not None),
                              frontier="" if session.frontier is None else f"{session.frontier[0]:.1f} {session.frontier[1]:.1f}",
                              goal="" if session.goal is None else f"{session.goal[0]:.1f} {session.goal[1]:.1f}",
                              directory=int(directory_s is not None)))
        if session.collisions + hits > seen_hits:     # say what each collision was, to tell walls from people
            seen_hits = session.collisions + hits
            r = sc.robot
            near = min((math.dist((r.x, r.y), (p.x, p.y)) for p in sc.crowd.everyone), default=math.inf)
            print(f"    collision at {sc.sup.getTime() - t0:.1f} s ({r.x:.1f}, {r.y:.1f}): wall {sc.plan.clearance_at(r.x, r.y):.2f} m, "
                  f"nearest person {near:.2f} m, {session.status()[:50]}", flush=True)
        if session.mode == "waiting" and directory_s is None:   # read a directory (or found none): pick
            directory_s = sc.sup.getTime() - t0
            path, hits = session.path_m, session.collisions
            session.go_to(name)
        if session.mode in ("arrived", "done"):
            break
    if trace is not None and steps:
        trace.parent.mkdir(parents=True, exist_ok=True)
        with trace.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0]))
            w.writeheader()
            w.writerows(steps)
    here = (sc.robot.x, sc.robot.y)
    error = min(math.dist(here, p) for p in at(layout, name))
    arrived = session.mode == "arrived" and error < 2.0
    return dict(arrived=int(arrived), time=sc.sup.getTime() - t0 if arrived else limit, path=path + session.path_m,
                collisions=hits + session.collisions, directory_s=directory_s if directory_s is not None else limit,
                seen=session.map.explored_share(sc.plan), places_read=len(session.reader.places()),
                stop_error=error, ended=session.mode,
                after_directory_s=(sc.sup.getTime() - t0 if arrived else limit) - (directory_s or 0.0),
                door_looks=looks["doors"], other_looks=looks["other"], looking_s=looks["s"], in_rooms_s=looks["rooms"])


def rooms_of(plan):
    """The building's rooms, from the true floor plan (for this report only, the chair never sees it):
    open floor split at doorways, the biggest region being the corridors and halls."""
    from webots.signs import Geometry
    return Geometry(plan)


def in_a_room(geo, x, y) -> bool:
    r, c = geo.px(x, y)
    rows, cols = geo.labels.shape
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    label = int(geo.labels[r, c])
    return label not in (0, geo.main)


def save(rows, out: Path) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default="fp_hospital")
    p.add_argument("--trips", type=int, default=10)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--limit", type=float, default=400.0, help="seconds per trip, finding the directory included (default 400)")
    p.add_argument("--read", choices=["text", "tags"], default="text")
    p.add_argument("--ocr-s", type=float, default=0.0,
                   help="as if reading a sign's words took this long, one at a time, as run.py's reading thread "
                        "does while the trip goes on, such as 0.2 (default 0: every sign read at once)")
    p.add_argument("--ways", nargs="+", default=["plan", "signs"])
    p.add_argument("--port", type=int, default=1236)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", help="results CSV (default data/signs/<world>_open.csv)")
    p.add_argument("--given-directory", action="store_true",
                   help="the chair is told the building's directory at the start, as if it had read one where "
                        "it stands, so only finding the place is measured")
    p.add_argument("--trace", action="store_true",
                   help="also save what the chair did every second of each signs trip, to data/signs/trace/")
    args = p.parse_args(argv)
    world = ROOT / "webots" / "worlds" / f"{args.world}.wbt"
    process = launch.start_webots(world, args.port, fast=True, headless=args.headless, log=ROOT / "webots" / "webots.log")
    launch.connect_to(args.port)
    try:
        run(world, args)
    finally:
        launch.stop_webots(process)


def run(world: Path, args) -> None:
    from baselines.drivers import make_driver
    from webots import signs
    from webots.bridge import WebotsScene
    from webots.open_world import Session
    scene = WebotsScene(world, animate=False, trail=False, stream=True)
    layout = signs.for_world(scene.sup, scene.plan, world.stem)
    driver = make_driver("assist", scene)
    session = Session(scene, driver, layout, read=args.read, wait_for_ocr=True, markers=False, ocr_s=args.ocr_s)
    if args.given_directory:                         # as if read: the true list, every entry with its wing
        listing = {name: wing for f in layout.faces if f.kind == "directory" for name, wing in f.rows}
        session.reader.directory = lambda: dict(listing)
        session.reader.have_directory = lambda: True
    plan_list = trips(layout, scene, args.trips, args.seed, random.Random(args.seed))
    rows = []
    out = Path(args.out) if args.out else ROOT / "data" / "signs" / f"{world.stem}_open.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed, name in plan_list:
        for way in args.ways:
            wall = time.time()
            r = with_plan(scene, driver, layout, seed, name, args.limit) if way == "plan" else \
                with_signs(session, seed, name, args.limit, layout,
                           ROOT / "data" / "signs" / "trace" / f"{world.stem}_{seed}.csv" if args.trace else None)
            r = dict(seed=seed, destination=name, way=way, **{k: round(v, 2) if isinstance(v, float) else v
                                                             for k, v in r.items()})
            rows.append(r)
            print(f"seed {seed} {name:22s} {way:6s} {'arrived' if r['arrived'] else 'NOT THERE':9s} "
                  f"{r['time']:6.1f} s {r['path']:6.1f} m  collisions {r['collisions']}"
                  + (f"  directory after {r['directory_s']:.0f} s, then {r['after_directory_s']:.0f} s with "
                     f"{r['door_looks']} door and {r['other_looks']} other looks ({r['looking_s']:.0f} s), "
                     f"{r['in_rooms_s']:.0f} s in rooms, "
                     f"explored {r['seen']:.0%}, read {r['places_read']} places, ended {r['ended']}"
                     if way == "signs" else "") + f"   ({time.time() - wall:.0f} s real)", flush=True)
            save(rows, out)                              # after every trip, so a stopped run keeps what it has
    print(f"\nFinding a named place, {world.stem}: {len(plan_list)} trips, {args.limit:.0f} s limit, OCR: {args.read}"
          f" ({args.ocr_s:g} s a sign)" + (", the directory given" if args.given_directory else ""))
    print(f"{'way':8s} {'arrived':>8s} {'time s':>8s} {'path m':>8s} {'collisions':>11s} {'seen':>6s} {'read':>6s}")
    for way in args.ways:
        sel = [r for r in rows if r["way"] == way]
        ok = [r for r in sel if r["arrived"]]
        mean = lambda k, s=ok: sum(r[k] for r in s) / len(s) if s else float("nan")
        extra = f" {mean('seen', sel):6.0%} {mean('places_read', sel):6.1f}" if way == "signs" else ""
        print(f"{way:8s} {len(ok) / len(sel):8.0%} {mean('time'):8.1f} {mean('path'):8.1f} "
              f"{sum(r['collisions'] for r in sel):11d}{extra}")
    found = sorted(r["directory_s"] for r in rows if r["way"] == "signs")
    if found:
        print(f"Directory read after: median {found[len(found) // 2]:.0f} s (of the time above)")
        sel = [r for r in rows if r["way"] == "signs"]
        after = sorted(r["after_directory_s"] for r in sel)
        print(f"From the directory to the place: median {after[len(after) // 2]:.0f} s, longest {after[-1]:.0f} s; "
              f"looks closer a trip: {sum(r['door_looks'] for r in sel) / len(sel):.1f} at doors, "
              f"{sum(r['other_looks'] for r in sel) / len(sel):.1f} at other signs, "
              f"{sum(r['looking_s'] for r in sel) / len(sel):.0f} s; in rooms {sum(r['in_rooms_s'] for r in sel) / len(sel):.0f} s")
    both = [(a, b) for a in rows for b in rows if a["seed"] == b["seed"] and a["way"] == "plan" and b["way"] == "signs"
            and a["arrived"] and b["arrived"]]
    if both:
        ratio = sorted(b["time"] / a["time"] for a, b in both)
        print(f"Trips both ways got there: {len(both)}; time without the plan / with it: median {ratio[len(ratio) // 2]:.1f}x")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
