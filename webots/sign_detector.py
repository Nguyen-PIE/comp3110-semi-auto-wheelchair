"""A sign detector trained on pictures from Webots that label themselves, and how it compares with tags.

    python webots/sign_detector.py collect           # take labelled pictures in four buildings and the depot
    python webots/sign_detector.py train             # train a small YOLO detector on them (CPU is fine)
    python webots/sign_detector.py evaluate          # how many signs it finds, by distance, against the tag reader

The tag reader (vision.py) needs a sign's tag to be at least about 10 pixels across, so it stops
seeing signs beyond about 10 m. A detector that has learned what a sign looks like (a coloured
panel with writing) could find them further away and without tags. The training pictures cost
nothing to label: Webots knows where every sign is (signs.py), so each sign face in view is
projected into the picture through the camera's own geometry.

collect   chair at random spots facing a random sign 1.5 to 12 m away, nobody around. A sign face is
          labelled if it faces the camera, nothing on the floor plan is between them and at least
          half of it is in the picture. Classes: plate, blade, direction, directory. Pictures from the
          hospital, the office, warehouse3 and the corridor are split 80/20 into training and
          validation; the depot (never trained on) is a test of a new building.
          Saved in data/signs/detector/ with data.yaml (the YOLO format).
train     fine-tunes YOLO11n (Ultralytics, AGPL-3.0; pip install ultralytics) from its COCO weights;
          the best weights go to models/sign_detector.pt.
evaluate  on the validation and depot pictures: share of labelled signs the detector finds
          (box overlap 0.5 or more) and the tag reader finds, by distance; the detector's precision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import launch  # noqa: E402

DATA = ROOT / "data" / "signs" / "detector"
MODEL = ROOT / "models" / "sign_detector.pt"
CLASSES = ["plate", "blade", "direction", "directory"]
TRAIN_WORLDS = ["fp_hospital", "fp_office", "fp_warehouse3", "corridor"]
TEST_WORLD = "fp_depot"
DIST_BINS = [(0, 4), (4, 7), (7, 10), (10, 14), (14, 30)]


def corners_of(face):
    from webots import signs
    k = signs.KINDS[face.kind]
    rx, ry = -math.sin(face.yaw), math.cos(face.yaw)          # the viewer's right on the face
    out = []
    for sx, sz in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
        out.append((face.x + rx * sx * k["w"] / 2, face.y + ry * sx * k["w"] / 2, face.z + sz * k["h"] / 2))
    return out


def project(points, pose, K):
    """World points -> pixels (u, v) and depth, through the chair's camera."""
    from webots.vision import CAMERA_PITCH, CAMERA_X, CAMERA_Z
    x0, y0, z0, h = pose
    c, s = math.cos(h), math.sin(h)
    cp, sp = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    out = []
    for X, Y, Z in points:
        dx, dy = X - x0, Y - y0
        fx, fy = c * dx + s * dy - CAMERA_X, -s * dx + c * dy
        fz = Z - z0 - CAMERA_Z
        ahead, up = fx * cp + fz * sp, -fx * sp + fz * cp   # through the camera's upward tilt
        cam = np.array([-fy, -up, ahead])                  # OpenCV camera: right, down, ahead
        if cam[2] <= 0.1:
            return None
        u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
        v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
        out.append((u, v, cam[2]))
    return out


def labels_for(layout, plan, pose, K, size):
    """YOLO boxes of every sign face in view: [(class, cx, cy, w, h normalised, distance, tag)]."""
    from webots.sign_eval import clear_line
    from webots.vision import CAMERA_X
    W, H = size
    x0, y0, _, h = pose
    cam_xy = (x0 + math.cos(h) * CAMERA_X, y0 + math.sin(h) * CAMERA_X)
    out = []
    for f in layout.faces:
        to_cam = (cam_xy[0] - f.x, cam_xy[1] - f.y)
        if to_cam[0] * math.cos(f.yaw) + to_cam[1] * math.sin(f.yaw) <= 0.05:
            continue                                              # its back is to the camera
        d = math.hypot(*to_cam)
        if d > 30:
            continue
        near = (f.x + math.cos(f.yaw) * 0.2, f.y + math.sin(f.yaw) * 0.2)
        if not clear_line(plan, cam_xy, near):
            continue
        pts = project(corners_of(f), pose, K)
        if pts is None:
            continue
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        box = (min(us), min(vs), max(us), max(vs))
        clipped = (max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H))
        area = (box[2] - box[0]) * (box[3] - box[1])
        inside = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
        if area <= 0 or inside / area < 0.5 or clipped[3] - clipped[1] < 5 or clipped[2] - clipped[0] < 5:
            continue
        cx, cy = (clipped[0] + clipped[2]) / 2 / W, (clipped[1] + clipped[3]) / 2 / H
        out.append((CLASSES.index(f.kind), cx, cy, (clipped[2] - clipped[0]) / W, (clipped[3] - clipped[1]) / H, d, f.tag))
    return out


def collect(args) -> None:
    """Every building, each in a process of its own (Webots lets a process connect only once)."""
    import subprocess
    if args.world:
        collect_world(args)
        return
    for old in ("images", "labels"):                   # pictures from an earlier run could be in another split
        shutil.rmtree(DATA / old, ignore_errors=True)
    meta = {}
    for world_name in TRAIN_WORLDS + [TEST_WORLD]:
        part = DATA / f"meta_{world_name}.json"
        part.unlink(missing_ok=True)
        # (the exit code is no guide: Webots' controller library exits with 1 when the simulator closes)
        subprocess.run([sys.executable, __file__, "collect", "--world", world_name, "--pictures", str(args.pictures),
                        "--port", str(args.port)] + (["--headless"] if args.headless else []))
        if not part.exists():
            sys.exit(f"Taking pictures in {world_name} failed (see above)")
        meta.update(json.loads(part.read_text()))
        part.unlink()
    (DATA / "meta.json").write_text(json.dumps(meta))
    (DATA / "data.yaml").write_text(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                    f"names: {dict(enumerate(CLASSES))}\n")
    counts = {s: sum(1 for m in meta.values() if m["split"] == s) for s in ("train", "val", "test")}
    print(f"Saved {sum(counts.values())} pictures ({counts}) and {sum(len(m['boxes']) for m in meta.values())} signs in {DATA}")


def collect_world(args) -> None:
    import cv2
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (DATA / kind / split).mkdir(parents=True, exist_ok=True)
    meta = {}
    for world_name in [args.world]:
        world = ROOT / "webots" / "worlds" / f"{world_name}.wbt"
        process = launch.start_webots(world, args.port, fast=True, headless=args.headless, log=ROOT / "webots" / "webots.log")
        launch.connect_to(args.port)
        try:
            from webots import signs
            from webots.bridge import WebotsScene
            from webots.sign_eval import views_of
            from webots.vision import SignReader
            scene = WebotsScene(world, animate=False, trail=False, stream=False)
            scene.reset(seed=10000)
            for p in scene.crowd.everyone:
                p.x, p.y = 1000.0, 1000.0
            scene.crowd.walking, scene.crowd.standing = [], []
            scene._draw_people(snap=True)                 # and their models with them
            layout = signs.for_world(scene.sup, scene.plan, world.stem)
            reader = SignReader(scene, layout, read="tags", every_s=0.1)
            rng = random.Random(world_name)
            n = args.pictures if world_name != TEST_WORLD else args.pictures // 2
            taken = 0
            while taken < n:
                face = rng.choice(layout.faces)
                spots = views_of(scene.plan, face, 1, rng)
                if not spots:
                    continue
                x, y, heading, _, _ = spots[0]
                heading += rng.uniform(-0.35, 0.35)
                scene.chair_node.getField("translation").setSFVec3f([x, y, 0.002])
                scene.chair_node.getField("rotation").setSFRotation([0, 0, 1, heading])
                scene.chair_node.resetPhysics()
                scene.hold_still()
                scene.hold_still()
                img, pose = reader.picture(), reader.chair_pose()
                boxes = labels_for(layout, scene.plan, pose, reader.K, reader.size)
                if not boxes:
                    continue
                split = "test" if world_name == TEST_WORLD else ("val" if rng.random() < 0.2 else "train")
                name = f"{world_name}_{taken:04d}"
                cv2.imwrite(str(DATA / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DATA / "labels" / split / f"{name}.txt").write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h, _, _ in boxes))
                meta[name] = dict(split=split, boxes=[dict(cls=c, box=[cx, cy, w, h], distance=round(d, 2), tag=t)
                                                      for c, cx, cy, w, h, d, t in boxes],
                                  tags=[t for t, _ in reader.detect(img)])
                taken += 1
            print(f"{world_name}: {taken} pictures", flush=True)
        finally:
            launch.stop_webots(process)
    (DATA / f"meta_{args.world}.json").write_text(json.dumps(meta))


def train(args) -> None:
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=str(DATA / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu",
                project=str(DATA / "runs"), name="train", exist_ok=True, workers=2, plots=False, verbose=False)
    best = DATA / "runs" / "train" / "weights" / "best.pt"
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL)
    print(f"Saved {MODEL}")


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def evaluate(args) -> None:
    from ultralytics import YOLO
    meta = json.loads((DATA / "meta.json").read_text())
    model = YOLO(str(MODEL))
    for split in ("val", "test"):
        names = [n for n, m in meta.items() if m["split"] == split]
        found = {b: [0, 0, 0] for b in DIST_BINS}         # labelled, detector found, tag reader found
        predicted = correct = 0
        for name in names:
            m = meta[name]
            result = model.predict(str(DATA / "images" / split / f"{name}.jpg"), imgsz=args.imgsz, conf=0.25,
                                   device="cpu", verbose=False)[0]
            preds = [(int(c), *map(float, b)) for c, b in zip(result.boxes.cls, result.boxes.xywhn)]
            predicted += len(preds)
            matched = set()
            for box in m["boxes"]:
                b = next(k for k in DIST_BINS if k[0] <= box["distance"] < k[1])
                found[b][0] += 1
                hit = [i for i, p in enumerate(preds) if p[0] == box["cls"] and _iou(p[1:], box["box"]) >= 0.5]
                if hit:
                    found[b][1] += 1
                    matched.add(hit[0])
                found[b][2] += box["tag"] in m["tags"]
            correct += len(matched)
        place = "validation (buildings trained on, new pictures)" if split == "val" else "the depot (never trained on)"
        total = [sum(v[i] for v in found.values()) for i in range(3)]
        print(f"\n{place}: {len(names)} pictures, {total[0]} signs; detector precision {correct / max(1, predicted):.0%}")
        print(f"{'distance':>10s} {'signs':>6s} {'detector':>9s} {'tags':>6s}")
        for (a, b), (n, det, tag) in found.items():
            if n:
                print(f"{f'{a}-{b} m':>10s} {n:6d} {det / n:9.0%} {tag / n:6.0%}")
        print(f"{'all':>10s} {total[0]:6d} {total[1] / max(1, total[0]):9.0%} {total[2] / max(1, total[0]):6.0%}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["collect", "train", "evaluate"])
    p.add_argument("--pictures", type=int, default=120, help="pictures per building (default 120; the depot half)")
    p.add_argument("--world", help=argparse.SUPPRESS)       # collect: one building (run by collect itself)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--weights", default="yolo11n.pt", help="starting weights (default YOLO11n, downloaded once)")
    p.add_argument("--port", type=int, default=1237)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args(argv)
    {"collect": collect, "train": train, "evaluate": evaluate}[args.step](args)


if __name__ == "__main__":
    main()
