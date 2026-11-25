"""Turn the SiT recordings (people walking around a robot) into training data for the walking model.

SiT ("Socially Interactive Trajectory", Hanyang University's SPA Lab, NeurIPS 2023 Datasets track,
https://spalaboratory.github.io/SiT/) was recorded from a Clearpath Husky robot driven through
crowds: corridors, hallways, a lobby, a cafeteria, a courtyard and some streets. Unlike ETH/UCY
(filmed from above, no robot) every recording has a robot in it, and the people react to it. That
is exactly what we want our simulated people to learn: how real people move around a wheelchair-
sized robot.

It reads any of these (it works out which one it was given):
  - the tracks SiT publish on GitHub (github.com/SPALaboratory/SiT-Dataset, folder
    prediction/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/trajectory_txt):
    <folder>/<Place>/<Place_N>/trajectory.txt, lines "frame class id x y yaw", robot = "husky";
  - the output of extract_from_drive.py (same format, as a folder or its .zip);
  - the full SiT download: <folder>/<Place>/<Place_N>/label_3d/<frame>.txt (one line per object:
    class, ID:<n>, h, l, w, x, y, z, rot) and ego_trajectory/<frame>.txt (4x4 pose, comma separated).

What it does:
  - keeps people of class "Pedestrian" (not Pedestrain_sitting [sic], cars, bikes, ...);
  - keeps every 4th frame (SiT is 10 frames a second; the walking model thinks in 0.4 s steps,
    as ETH/UCY);
  - adds the robot as track 0, an extra "person", the same way webots/learned_walk.py adds the
    wheelchair, so the model learns how people react to it;
  - splits a person's track where they were not seen for a while (Trajectron++ needs unbroken
    tracks), and drops tracks shorter than 2 steps;
  - writes the ETH/UCY text format (frame, track, x, y) into Trajectron++'s raw data folder,
    using SiT's own train / val / test split of scenes (sit_split.json: 37 / 14 / 14 scenes, copied
    from SiT's repository). Trajectron++'s own process_data.py then prepares it (train.sh does this).

With --with-eth the ETH/UCY training recordings (all places but zara1, as for our current model)
are added to SiT's training and validation scenes, as "sitmix".

SiT's licence is CC BY-NC-ND 4.0 (non-commercial, no redistribution of the data): fine for a
university project, but keep the dataset out of the repository and only share the trained model.

    python webots/walk_model/sit_to_trajectron.py <SiT tracks: folder or .zip> [--with-eth]
    bash webots/walk_model/train.sh 100 sit          # (or sitmix) train on it
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "third_party" / "Trajectron-plus-plus" / "experiments" / "pedestrians" / "raw"
SPLIT = Path(__file__).resolve().parent / "sit_split.json"
SIT_HZ = 10                 # SiT's frames per second
EVERY = 4                   # keep every 4th: 0.4 s, the walking model's step
ROBOT_ID = 0                # the robot's track number (people start at 1) ...
ROBOT_EXTRA = 900_000       # ... and, if its pose is missing for a while, the pieces after the gap
PERSON = "Pedestrian"       # SiT also labels Pedestrain_sitting (sic), cars, bikes, ...: left out


def is_robot(track_id) -> bool:
    return int(track_id) == ROBOT_ID or int(track_id) > ROBOT_EXTRA


# ---- reading: each scene becomes {frame: ([(sit_id, x, y), ...], robot (x, y) or None)} ----------

def read_robot(text: str) -> tuple[float, float]:
    """ego_trajectory/<frame>.txt: one line, a 4x4 pose matrix, comma separated."""
    m = np.array([float(v) for v in text.split("\n")[0].split(",") if v.strip()]).reshape(4, 4)
    return float(m[0, 3]), float(m[1, 3])


def read_people(text: str) -> list[tuple[int, float, float]]:
    """label_3d/<frame>.txt: one line per object: class, ID:<n>, h, l, w, x, y, z, rot."""
    people = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 9 and f[0] == PERSON:
            people.append((int(f[1].split(":")[1]), float(f[5]), float(f[6])))
    return people


def frames_from_label_dirs(scene_dir: Path) -> dict:
    labels = scene_dir / "label_3d"
    out = {}
    for p in labels.glob("*.txt"):
        ego = scene_dir / "ego_trajectory" / p.name
        out[int(p.stem)] = (read_people(p.read_text()),
                            read_robot(ego.read_text()) if ego.exists() and ego.read_text().strip() else None)
    if not out:
        raise FileNotFoundError(f"no label_3d/*.txt in {scene_dir}")
    return out


def frames_from_trajectory_txt(text: str) -> dict:
    """trajectory.txt: "frame class id x y yaw" per line (SiT's GitHub trajectory_txt format)."""
    out: dict[int, tuple[list, object]] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        frame = int(float(f[0]))
        people, robot = out.get(frame, ([], None))
        if f[1].lower() == "husky":
            robot = (float(f[3]), float(f[4]))
        elif f[1] == PERSON:
            people.append((int(f[2]), float(f[3]), float(f[4])))
        out[frame] = (people, robot)
    return out


def find_scenes(src: Path) -> dict[str, callable]:
    """Scene name (e.g. Corridor_3) -> function returning its frames, for any of the three inputs."""
    scenes = {}
    if src.is_file() and src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        for name in z.namelist():
            parts = name.split("/")
            if parts[-1] == "trajectory.txt" and len(parts) >= 2:
                scenes[parts[-2]] = lambda n=name: frames_from_trajectory_txt(z.read(n).decode())
        return scenes
    for t in src.rglob("trajectory.txt"):
        scenes[t.parent.name] = lambda t=t: frames_from_trajectory_txt(t.read_text())
    for lab in src.rglob("label_3d"):
        if lab.is_dir() and lab.parent.name not in scenes:
            scenes[lab.parent.name] = lambda d=lab.parent: frames_from_label_dirs(d)
    return scenes


# ---- converting ---------------------------------------------------------------------------------

def convert_frames(frames: dict, robot: bool = True) -> tuple[list[tuple[int, int, float, float]], dict]:
    """Rows (frame, track, x, y) in the ETH/UCY format: frames numbered 0, 10, 20, ... every 0.4 s."""
    order = sorted(frames)
    first = order[0]
    kept = [f for f in order if (f - first) % EVERY == 0]
    seen: dict[int, list[tuple[int, float, float]]] = {}      # SiT id -> (step, x, y)
    robot_track = []
    for step, f in enumerate(kept):
        people, pose = frames[f]
        for pid, x, y in people:
            seen.setdefault(pid, []).append((step, x, y))
        if robot and pose is not None:
            robot_track.append((step, *pose))
    rows, next_id, dropped, robot_pieces = [], 1, 0, 0
    tracks = ([("robot", robot_track)] if robot and robot_track else []) + [("person", t) for t in seen.values()]
    for kind, track in tracks:
        track.sort()
        pieces, piece = [], [track[0]]
        for a, b in zip(track, track[1:]):
            if b[0] != a[0] + 1:                              # not seen for a while: a new piece
                pieces.append(piece)
                piece = []
            piece.append(b)
        pieces.append(piece)
        for piece in pieces:
            if len(piece) < 2:
                dropped += 1
                continue
            if kind == "robot":                               # 0, then (rarely, after a gap) 900001, ...
                tid = ROBOT_ID if robot_pieces == 0 else ROBOT_EXTRA + robot_pieces
                robot_pieces += 1
            else:
                tid, next_id = next_id, next_id + 1
            rows += [(step * 10, tid, x, y) for step, x, y in piece]
    people = len({r[1] for r in rows if not is_robot(r[1])})
    return sorted(rows), {"steps": len(kept), "people": people, "dropped": dropped,
                          "robot": any(r[1] == ROBOT_ID for r in rows)}


def convert_scene(scene_dir: Path, robot: bool = True):
    """A scene folder with label_3d/ (+ ego_trajectory/) or trajectory.txt."""
    t = scene_dir / "trajectory.txt"
    frames = frames_from_trajectory_txt(t.read_text()) if t.exists() else frames_from_label_dirs(scene_dir)
    return convert_frames(frames, robot)


def write(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x, y in rows:
            f.write(f"{frame}\t{tid}\t{x:.3f}\t{y:.3f}\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sit", type=Path, help="SiT tracks: trajectory_txt folder, sit_trajectories(.zip), or the full download")
    p.add_argument("--with-eth", action="store_true",
                   help="also add the ETH/UCY training recordings (all but zara1), as source 'sitmix'")
    p.add_argument("--no-robot", action="store_true", help="leave the robot out (people only)")
    p.add_argument("--raw", type=Path, default=RAW, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    split = json.loads(SPLIT.read_text())
    source = "sitmix" if args.with_eth else "sit"
    out = args.raw / source
    if not args.sit.exists():
        sys.exit(f"Not found: {args.sit}")
    scenes = find_scenes(args.sit)
    if not scenes:
        sys.exit(f"No scenes found in {args.sit}: expected <Place>/<Place_N>/trajectory.txt "
                 f"or <Place>/<Place_N>/label_3d/")
    if out.exists():
        shutil.rmtree(out)
    total = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    missing = []
    for part, names in split.items():
        for name in names:
            if name not in scenes:
                missing.append(name)
                continue
            rows, info = convert_frames(scenes[name](), robot=not args.no_robot)
            write(rows, out / part / f"sit_{name}.txt")
            total[part][0] += 1
            total[part][1] += info["people"]
            print(f"  {part:5} {name:26} {info['steps'] * EVERY / SIT_HZ:6.0f} s  {info['people']:4} people"
                  + ("" if info["robot"] else "  (no robot pose found)"))
    if args.with_eth:
        eth = args.raw / "zara1"
        if not (eth / "train").is_dir():
            sys.exit(f"ETH/UCY recordings not found in {eth} (run webots/walk_model/get_trajectron.sh first)")
        for part in ("train", "val"):
            for f in (eth / part).glob("*.txt"):
                shutil.copy(f, out / part / f"eth_{f.name}")
    if missing:
        print(f"  ({len(missing)} of the {sum(map(len, split.values()))} scenes in SiT's split are not in {args.sit}, "
              f"e.g. {', '.join(missing[:3])})")
    for part, (n, people) in total.items():
        print(f"{part}: {n} SiT scenes, {people} people")
    print(f"Written to {out}. Next:  bash webots/walk_model/train.sh 100 {source}")


if __name__ == "__main__":
    main()
