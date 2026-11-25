"""Download the SiT dataset from its public Google Drive folder to a big disk, resuming where it stopped.

    python scripts/download_sit.py "/Volumes/2TB EXT Hardrive/Pierre's stuff/comp3110/SiT"
    python scripts/download_sit.py DEST --only Corridor_2        # just the files whose path contains this
    python scripts/download_sit.py DEST --indoor-only            # Corridor, Hallway, Lobby, Cafeteria

SiT ("Socially Interactive Pedestrian Trajectory Dataset", https://spalaboratory.github.io/SiT/) is
about 3 to 7.5 GB per scene zip, 66 files in all (roughly 300 GB), each zip holding a scene's camera
images, lidar and labels. Only the labels are needed for scripts/sit_tracks.py, but Drive cannot
hand out part of a zip, so the whole zip comes down; extract the labels and delete the zip afterwards
if space matters. Do not put the data in the repository (data/sit/ is ignored by Git); the licence is
share-alike on the project page and no-derivatives on its GitHub, so do not republish it.

Needs `pip install gdown`. Google limits how often a public file can be downloaded; a file that hits
the limit is reported and skipped, and a later run picks it up again (finished files are skipped,
half-finished ones resume).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gdown

FOLDER_ID = "1t55r1kpxlLQP458fS-DjWvOq8aLPkFWI"
INDOOR = ("Corridor", "Hallway", "Lobby", "Cafeteria")


def order(path: str) -> tuple:
    """Split lists first, then the indoor scenes (what the hospital sim is like), then the outdoor ones."""
    top = path.split("/")[0]
    return (0 if top == "ImageSets" else 1 if top in INDOOR else 2, path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dest", type=Path, help="folder to put the dataset in (made if missing)")
    p.add_argument("--only", help="only files whose path contains this text")
    p.add_argument("--indoor-only", action="store_true", help="skip the outdoor scenes")
    p.add_argument("--list", action="store_true", help="list the files and stop")
    args = p.parse_args(argv)

    listing = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True)
    files = sorted(((f.id, f.path) for f in listing), key=lambda f: order(f[1]))
    if args.only:
        files = [f for f in files if args.only in f[1]]
    if args.indoor_only:
        files = [f for f in files if f[1].split("/")[0] in INDOOR or f[1].startswith("ImageSets")]
    if args.list:
        print("\n".join(path for _, path in files))
        return
    args.dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for n, (fid, path) in enumerate(files, 1):
        target = args.dest / path
        done = target.with_name(target.name + ".done")
        if done.exists() and target.exists():
            print(f"[{n}/{len(files)}] have {path}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{n}/{len(files)}] {path}", flush=True)
        started = time.time()
        try:
            got = gdown.download(id=fid, output=str(target), quiet=False, resume=True)
        except Exception as error:  # a quota or network error: report it, go on to the next file
            got = None
            print(f"   failed: {error}", flush=True)
        if got:
            done.write_text(f"{target.stat().st_size}\n")
            print(f"   {target.stat().st_size / 1e9:.2f} GB in {(time.time() - started) / 60:.1f} min", flush=True)
        else:
            failed.append(path)
    if failed:
        print(f"\n{len(failed)} not downloaded (try again later):\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nAll downloaded.")


if __name__ == "__main__":
    main()
