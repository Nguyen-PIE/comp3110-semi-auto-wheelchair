"""Turn a picture of a floor plan into a map the simulator can use.

Give it an image (PNG or JPG; export a PDF page as PNG first) where walls are
dark lines on a light background, and tell it how big the building is:

    python scripts/import_floorplan.py level2.png --name level2 --width-m 42
    python scripts/demo.py --scenario maps/level2.yaml

This writes maps/level2.png and maps/level2.yaml (the ROS map format, the same
as the hospital, office and warehouse maps).

How the picture is read:
  - Pixels darker than --threshold are walls. Use --invert for light lines on
    a dark background.
  - Specks smaller than --min-blob-m2 are removed. Text and symbols are not:
    erase them first (see the tips below).
  - Open floor that touches the edge of the picture is treated as outside the
    building and left out, so the chair and people stay indoors. Gaps under
    60 cm in the outside wall are ignored, but a wide open entrance lets the
    outside in: draw a line across it, or use --keep-outside on purpose.

Tips for a good result:
  - Crop the picture to the building, and erase text, furniture you do not
    want, and door-swing arcs (otherwise they become walls across doorways).
  - Get the scale right: measure one known length (a corridor, a door is
    about 0.9 m) and check the printed size. Doors narrower than about 0.9 m
    in the map will be too narrow for the chair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.floorplan import load_floorplan  # noqa: E402

TARGET_RES = 0.05            # metres per pixel in the saved map, like the wheelchair-navi maps
SEAL_M = 0.3                 # gaps in the outside wall narrower than 2 x this are treated as closed
FREE_PX, WALL_PX, UNKNOWN_PX = 254, 0, 205   # ROS map_server colours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="floor-plan picture (PNG or JPG)")
    parser.add_argument("--name", help="map name (default: the picture's file name)")
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument("--width-m", type=float, help="real width of the whole picture in metres")
    scale.add_argument("--metres-per-pixel", type=float, help="scale of the picture")
    parser.add_argument("--threshold", type=int, default=128, help="0-255; darker pixels are walls (default 128)")
    parser.add_argument("--invert", action="store_true", help="the plan has light lines on a dark background")
    parser.add_argument("--thicken-cm", type=float, default=0.0, help="make walls this much thicker (fills small gaps)")
    parser.add_argument("--min-blob-m2", type=float, default=0.05, help="remove dark specks smaller than this (m²)")
    parser.add_argument("--keep-outside", action="store_true", help="keep open space outside the building")
    parser.add_argument("--out", default=str(ROOT / "maps"), help="folder to save the map in (default: maps/)")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Could not read {args.image}. Use a PNG or JPG (export PDF pages as PNG first).")
    name = args.name or Path(args.image).stem
    res = args.metres_per_pixel or args.width_m / image.shape[1]

    walls = (image > args.threshold) if args.invert else (image < args.threshold)

    # Resample to 5 cm per pixel. Shrinking keeps thin lines: a block is a wall
    # if a quarter or more of it is dark.
    factor = res / TARGET_RES
    if abs(factor - 1.0) > 0.01:
        size = (max(1, round(image.shape[1] * factor)), max(1, round(image.shape[0] * factor)))
        if factor < 1:
            walls = cv2.resize(walls.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.25
        else:
            walls = cv2.resize(walls.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        res = res * image.shape[1] / size[0]

    # Remove specks such as dimension marks and stray pixels.
    min_px = args.min_blob_m2 / (res * res)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(walls.astype(np.uint8), connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < min_px)[0]
    walls[np.isin(labels, small[small != 0])] = False

    if args.thicken_cm > 0:
        k = max(1, int(round(args.thicken_cm / 100 / res)))
        walls = cv2.dilate(walls.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0

    # Open floor connected to the picture's edge is outside the building.
    out = np.full(walls.shape, FREE_PX, dtype=np.uint8)
    out[walls] = WALL_PX
    if not args.keep_outside:
        # Find it with walls temporarily thickened by SEAL_M, so small gaps in the
        # outside wall (drawing errors, thin glass doors) do not let it leak in.
        r = max(1, int(round(SEAL_M / res)))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sealed = cv2.dilate(walls.astype(np.uint8), disc) > 0
        n, labels = cv2.connectedComponents((~sealed).astype(np.uint8), connectivity=4)
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        outside = np.isin(labels, edge[edge != 0])
        bigger = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))
        outside = (cv2.dilate(outside.astype(np.uint8), bigger) > 0) & ~walls
        inside_m2 = float((~walls & ~outside).sum()) * res * res
        if inside_m2 < 1.0:
            print("Warning: no enclosed floor found (the outside wall probably has a gap), "
                  "so outside space is kept. Draw a line across open entrances to exclude it.")
        else:
            out[outside] = UNKNOWN_PX

    # Pad with a border of "unknown" so the building never touches the map edge.
    pad = int(round(0.5 / res))
    out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=UNKNOWN_PX)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path, yaml_path = out_dir / f"{name}.png", out_dir / f"{name}.yaml"
    cv2.imwrite(str(png_path), out)
    yaml_path.write_text(yaml.safe_dump({
        "image": png_path.name,
        "resolution": float(round(res, 6)),
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }, sort_keys=False))

    height_m, width_m = out.shape[0] * res, out.shape[1] * res
    print(f"Saved {yaml_path} and {png_path.name}: {width_m:.1f} x {height_m:.1f} m at {res * 100:.1f} cm per pixel")
    try:
        load_floorplan.cache_clear()
        plan = load_floorplan(yaml_path)
        reachable_m2 = len(plan.spawn_cells) * 0.25 * 0.25
        print(f"The chair can reach about {reachable_m2:.0f} m² of open floor (in one connected area).")
        if reachable_m2 < 30:
            print("That is very little. Check the scale (--width-m) and that doors are drawn as gaps.")
    except ValueError as exc:
        print(f"Warning: {exc}. Check the scale (--width-m), --threshold, and that doors are drawn as gaps.")
    print(f"\nTry it:\n    python scripts/demo.py --scenario {yaml_path.relative_to(ROOT) if yaml_path.is_relative_to(ROOT) else yaml_path}")


if __name__ == "__main__":
    main()
