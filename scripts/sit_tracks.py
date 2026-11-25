"""Turn the SiT dataset's pedestrian labels into a table of walking tracks, for the simulator.

SiT ("Socially Interactive Pedestrian Trajectory Dataset", Bae et al., NeurIPS 2023 Datasets and
Benchmarks; https://spalaboratory.github.io/SiT/) was recorded from a robot driving through
crowds, with 3D boxes and a track id for every person, 10 times a second. Access is by a Google
Form on the project page, so the data is not in this repository (data/sit/ is ignored by Git;
the licence is share-alike on the project page and no-derivatives on its GitHub, so do not
republish it). This script reads the labels you download and writes

    data/sit/tracks.csv       id,t,x,y      one row per person per frame, in metres and seconds

which webots/replay.py plays back in the simulator (`run.py --replay data/sit/tracks.csv`) and
scripts/fit_behaviour.py fits walking speeds, stops and groups from
(`--format table --columns id,t,x,y`).

    python scripts/sit_tracks.py path/to/SiT/labels --out data/sit/tracks.csv
    python scripts/sit_tracks.py "/Volumes/2TB EXT Hardrive/Pierre's Stuff/comp3110/SiT" --out data/sit/tracks.csv
    python scripts/sit_tracks.py path/to/SiT/labels --poses scene_poses.csv --out data/sit/tracks.csv

Quickest input, no 300 GB download: SiT publish every scene's tracks (27 MB, all 65 scenes) on their
GitHub. Fetch just that folder and point this script at it:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/SPALaboratory/SiT-Dataset.git /tmp/sit
    git -C /tmp/sit sparse-checkout set prediction/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/trajectory_txt
    python scripts/sit_tracks.py /tmp/sit/prediction --out data/sit/tracks.csv

Input: a folder with one text file per frame (searched recursively; the digits in the file name
are the frame number, the folder name is the scene), one object per line, as the SiT GitHub page
describes the 3D labels:  class  track_id  h  l  w  x  y  z  rot  (further columns ignored).
Only the pedestrian class is kept.

Checked on SiT's Corridor_1: the label_3d boxes are already in a fixed frame (seated people sit
still in them), so no pose correction is needed. For other data whose boxes are in a moving
robot's frame, `--poses` takes the robot's motion out: a CSV with
columns `scene,frame,x,y,yaw` (or `frame,x,y,yaw` for a single scene) of the robot's pose in a
fixed frame (for example from SiT's RTK or odometry), and each box is moved into that frame.
Without it the tracks are left in the robot's frame and a warning says so.

Real SiT lines look like `Pedestrian Pedestrian:1 h l w x y z rot`; seated people are the class
`Pedestrain_sitting` (sic) and are left out. Run it on one scene first and look at the summary it prints (about
1.2 m/s for walkers indoors) before trusting the rest.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 10.0                       # SiT's sensors run at 10 Hz
PEDESTRIAN = {"pedestrian", "person", "ped"}


def _frames(folder: Path):
    """(scene, frame, text) for each label file: loose text files under a folder, or label_3d/N.txt in the
    scene zips (Corridor_1.zip, ...) of a SiT download, read without unpacking the images."""
    for zpath in sorted(folder.rglob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if "/label_3d/" in "/" + name and name.endswith(".txt"):
                    yield zpath.stem, int(Path(name).stem), z.read(name).decode()
    for path in sorted(folder.rglob("*.txt")):
        if path.name == "trajectory.txt":       # SiT's published tracks: read in read_labels
            continue
        digits = re.findall(r"\d+", path.stem)
        if digits and "ImageSets" not in path.parts:
            yield path.parent.relative_to(folder).as_posix() or folder.name, int(digits[-1]), path.read_text()


def read_labels(folder: Path, hz: float = HZ) -> pd.DataFrame:
    """Every pedestrian in every frame file under `folder`: columns scene, frame, id, x, y."""
    rows = []
    for path in sorted(folder.rglob("trajectory.txt")):
        # SiT's own published tracks (github.com/SPALaboratory/SiT-Dataset, folder prediction/
        # Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/trajectory_txt): one file per
        # scene, lines "frame class id x y yaw", the same labels moved onto SiT's scene map.
        scene = path.parent.name
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].lower() in PEDESTRIAN:
                rows.append((scene, int(parts[0]), f"{scene}:ID:{parts[2]}", float(parts[3]), float(parts[4]), 0.0))
    for scene, frame, text in _frames(folder):
        for line in text.splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) < 8 or parts[0].lower() not in PEDESTRIAN:
                continue
            try:
                rows.append((scene, frame, f"{scene}:{parts[1]}", float(parts[5]), float(parts[6]), float(parts[7])))
            except ValueError:
                continue
    table = pd.DataFrame(rows, columns=["scene", "frame", "id", "x", "y", "z"])
    table["t"] = table["frame"] / hz
    return table


def to_fixed_frame(table: pd.DataFrame, poses: pd.DataFrame) -> pd.DataFrame:
    """Move each box from the robot's frame into the fixed frame the robot's poses are given in."""
    if "scene" not in poses.columns:
        poses = poses.assign(scene=table["scene"].iloc[0] if len(table) else "")
    merged = table.merge(poses[["scene", "frame", "x", "y", "yaw"]], on=["scene", "frame"], suffixes=("", "_robot"))
    c, s = np.cos(merged["yaw"]), np.sin(merged["yaw"])
    merged["x"], merged["y"] = (merged["x_robot"] + c * merged["x"] - s * merged["y"],
                                merged["y_robot"] + s * merged["x"] + c * merged["y"])
    return merged[["scene", "frame", "id", "x", "y", "t"]]


def summary(tracks: pd.DataFrame) -> str:
    """Speed and length of the tracks, to check they look like people walking."""
    speeds, lengths = [], []
    for _, g in tracks.sort_values("t").groupby("id"):
        if len(g) < 3:
            continue
        step = np.hypot(np.diff(g["x"].to_numpy()), np.diff(g["y"].to_numpy())) / np.diff(g["t"].to_numpy())
        speeds.append(float(np.median(step)))
        lengths.append(float(g["t"].iloc[-1] - g["t"].iloc[0]))
    if not speeds:
        return "no tracks of 3 or more frames"
    return (f"{len(speeds)} tracks, {len(tracks)} rows; median walking speed {np.median(speeds):.2f} m/s "
            f"(10th to 90th percentile {np.percentile(speeds, 10):.2f} to {np.percentile(speeds, 90):.2f}); "
            f"median length {np.median(lengths):.1f} s")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", type=Path, help="folder of per-frame label text files")
    p.add_argument("--out", type=Path, default=Path("data/sit/tracks.csv"))
    p.add_argument("--poses", type=Path, help="CSV of the robot's pose per frame: scene,frame,x,y,yaw")
    p.add_argument("--hz", type=float, default=HZ, help="frames per second (default 10)")
    p.add_argument("--min-seconds", type=float, default=2.0, help="drop tracks shorter than this (default 2)")
    args = p.parse_args(argv)

    table = read_labels(args.labels, args.hz)
    if table.empty:
        sys.exit(f"No pedestrian labels found under {args.labels}. Expected text files with lines "
                 "'Pedestrian track_id h l w x y z rot' (see the top of this file).")
    if args.poses:
        table = to_fixed_frame(table, pd.read_csv(args.poses))
    else:
        print("Note: no --poses; SiT's 3D labels are in a fixed frame already (seated people have zero speed "
              "in them, checked on Corridor_1), so none is needed for SiT itself.", file=sys.stderr)
    span = table.groupby("id")["t"].agg(lambda t: t.max() - t.min())
    table = table[table["id"].isin(span[span >= args.min_seconds].index)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.sort_values(["id", "t"])[["id", "t", "x", "y"]].to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {summary(table)}")
    print(f"Next: python scripts/fit_behaviour.py {args.out} --format table --columns id,t,x,y")


if __name__ == "__main__":
    main()
