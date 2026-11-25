"""How well the chair reads signs: put its camera in front of each sign at different distances and angles.

    python webots/sign_eval.py                       # the hospital, 8 views of every sign face
    python webots/sign_eval.py --world corridor --views 12
    python webots/sign_eval.py --headless            # Linux servers without a screen

For each sign face (signs.py) the chair is put, with nobody around, at random spots in front of it:
1.5 to 12 m away and up to 70 degrees to the side, on open floor with nothing between it and the
sign, facing the sign. One picture from each spot, read once (no voting), is compared with the true
answer from the building's list of signs:
    detected        the face's tag was found
    position error  where vision.py put the sign, against where it is (metres)
    facing error    the way it faces, against the truth (degrees)
    read            the words (plates and blade signs), every row's words and arrow (direction signs), or
                    "You are here" and every entry with its wing (directories) came out exactly right
                    (letters and digits, ignoring case and spacing)
    wrong tags      tags found that were not there, or put more than 1.5 m from where they are
Results by distance and by angle, for the whole building, and saved to data/signs/<world>_signs.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import launch  # noqa: E402

DISTANCES = (1.5, 12.0)
MAX_ANGLE = math.radians(70)
DIST_BINS = [(1.5, 3), (3, 5), (5, 7), (7, 9), (9, 12)]
ANGLE_BINS = [(0, 20), (20, 40), (40, 55), (55, 70)]


def clear_line(plan, a, b, step=0.1) -> bool:
    n = max(1, int(math.dist(a, b) / step))
    return all(plan.clearance_at(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) > 0.05
               for i in range(1, n))


def views_of(plan, face, n, rng):
    """Up to n spots (x, y, heading, distance, angle) the chair could see the face from."""
    out, tries = [], 0
    while len(out) < n and tries < 40 * n:
        tries += 1
        d = rng.uniform(*DISTANCES)
        off = rng.uniform(-MAX_ANGLE, MAX_ANGLE)
        a = face.yaw + off
        x, y = face.x + math.cos(a) * d, face.y + math.sin(a) * d
        if plan.clearance_at(x, y) < 0.5 or not clear_line(plan, (x, y), (face.x, face.y)):
            continue
        out.append((x, y, math.atan2(face.y - y, face.x - x), d, abs(math.degrees(off))))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default="fp_hospital")
    p.add_argument("--views", type=int, default=8, help="spots per sign face (default 8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=1235)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args(argv)
    world = ROOT / "webots" / "worlds" / f"{args.world}.wbt"
    process = launch.start_webots(world, args.port, fast=True, headless=args.headless, log=ROOT / "webots" / "webots.log")
    launch.connect_to(args.port)
    try:
        rows = measure(world, args)
    finally:
        launch.stop_webots(process)
    report(rows, args.world)


def measure(world: Path, args) -> list[dict]:
    from webots import signs, vision
    from webots.bridge import WebotsScene
    scene = WebotsScene(world, animate=False, trail=False, stream=False)
    scene.reset(seed=10000)
    for person in scene.crowd.everyone:              # nobody in the way: this measures reading, not crowds
        person.x, person.y = 1000.0, 1000.0
    scene.crowd.walking, scene.crowd.standing = [], []
    scene._draw_people(snap=True)                 # and their models with them
    layout = signs.for_world(scene.sup, scene.plan, world.stem)
    reader = vision.SignReader(scene, layout, read="text", every_s=0.1, wait_for_ocr=True)
    truth = layout.by_tag()
    rng = random.Random(args.seed)
    rows = []
    faces = list(layout.faces)
    for i, face in enumerate(faces):
        for x, y, heading, d, angle in views_of(scene.plan, face, args.views, rng):
            scene.chair_node.getField("translation").setSFVec3f([x, y, 0.002])
            scene.chair_node.getField("rotation").setSFRotation([0, 0, 1, heading])
            scene.chair_node.resetPhysics()
            scene.hold_still()
            scene.hold_still()
            img = reader.picture()
            pose = reader.chair_pose()
            detections = reader.detect(img)
            row = dict(tag=face.tag, kind=face.kind, distance=round(d, 2), angle=round(angle, 1), detected=0,
                       position_error=None, facing_error=None, read=None, text="", wrong_tags=0)
            for tag, corners in detections:
                place = reader.locate(vision.KIND_OF[tag], corners, pose)
                other = truth.get(tag)
                if other is None or (place and math.dist(place[:2], (other.x, other.y)) > 1.5):
                    row["wrong_tags"] += 1
                if tag != face.tag or place is None:
                    continue
                row["detected"] = 1
                row["position_error"] = round(math.dist(place[:2], (face.x, face.y)), 3)
                row["facing_error"] = round(abs(math.degrees(math.atan2(math.sin(place[3] - face.yaw),
                                                                        math.cos(place[3] - face.yaw)))), 1)
                least = vision.READ_MIN_TAG_PX_DIRECTORY if face.kind == "directory" else vision.READ_MIN_TAG_PX
                if reader.whole(face.kind, corners) and vision._side_px(corners) >= least:
                    _, _, result = reader._work((tag, face.kind, reader.rectify(img, face.kind, corners)))
                    if result[0] == "directory":
                        _, here, cells = result
                        got = sorted((vision.normalise(n), vision.normalise(w)) for _, n, w, _ in cells)
                        want = sorted((vision.normalise(n), vision.normalise(w)) for n, w in face.rows)
                        row["read"] = int(got == want and here == face.here)
                        row["text"] = f"{here} | " + " | ".join(f"{n} {w}" for _, n, w, _ in sorted(cells))
                    elif result[0] == "rows":
                        got = [(vision.normalise(t), a) for t, a in result[1]]
                        want = [(vision.normalise(t), a) for t, a in face.rows]
                        row["read"] = int(got == want)
                        row["text"] = " | ".join(f"{t} {a}" for t, a in result[1])
                    else:
                        row["read"] = int(vision.normalise(result[1]).replace(" ", "") ==
                                          vision.normalise(face.place).replace(" ", ""))
                        row["text"] = result[1]
                else:
                    row["read"] = 0
            rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(faces)} sign faces", flush=True)
    return rows


def report(rows: list[dict], world: str) -> None:
    out = ROOT / "data" / "signs" / f"{world}_signs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def line(name, sel):
        n = len(sel)
        if not n:
            return f"{name:16s} {0:5d}"
        det = [r for r in sel if r["detected"]]
        err = sorted(r["position_error"] for r in det)
        face = sorted(r["facing_error"] for r in det)
        read = sum(r["read"] or 0 for r in sel)
        return (f"{name:16s} {n:5d} {len(det) / n:9.0%} {(err[len(err) // 2] if err else float('nan')):10.2f} "
                f"{(err[int(0.9 * (len(err) - 1))] if err else float('nan')):8.2f} "
                f"{(face[len(face) // 2] if face else float('nan')):9.1f} {read / n:7.0%} "
                f"{sum(r['wrong_tags'] for r in sel):7d}")

    head = (f"{'':16s} {'views':>5s} {'detected':>9s} {'pos err m':>10s} {'90% m':>8s} {'facing':>9s} {'read':>7s} "
            f"{'wrong':>7s}")
    print(f"\nSign reading, {world}: one picture per view, read once (median errors; 90% = 90th percentile)")
    print(head)
    print(line("all", rows))
    for kind in ("plate", "blade", "direction", "directory"):
        print(line(kind, [r for r in rows if r["kind"] == kind]))
    print("by distance")
    for a, b in DIST_BINS:
        print(line(f"  {a:g}-{b:g} m", [r for r in rows if a <= r["distance"] < b]))
    print("by angle to the sign")
    for a, b in ANGLE_BINS:
        print(line(f"  {a}-{b} deg", [r for r in rows if a <= r["angle"] < b or (b == 70 and r["angle"] == 70)]))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
