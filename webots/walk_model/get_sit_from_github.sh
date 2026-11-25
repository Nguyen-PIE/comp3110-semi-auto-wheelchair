#!/usr/bin/env bash
# Download only SiT's published walking tracks (about 27 MB, 65 scenes) from SiT's own GitHub
# repository, straight onto this machine. No hard drive, no Google Form download, no copying needed.
#     bash get_sit_from_github.sh [where to put them, default ~/sit_data]
# Afterwards:  python webots/walk_model/sit_to_trajectron.py ~/sit_data/trajectory_txt
set -euo pipefail
DEST="${1:-$HOME/sit_data}"
TRACKS="prediction/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/trajectory_txt"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Fetching only the tracks folder from github.com/SPALaboratory/SiT-Dataset"
git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/SPALaboratory/SiT-Dataset.git "$TMP/sit"
git -C "$TMP/sit" sparse-checkout set "$TRACKS"

mkdir -p "$DEST"
rm -rf "$DEST/trajectory_txt"
cp -r "$TMP/sit/$TRACKS" "$DEST/trajectory_txt"
cp "$TMP/sit/LICENSE" "$DEST/LICENSE-code" 2>/dev/null || true

n=$(find "$DEST/trajectory_txt" -name trajectory.txt | wc -l)
echo "==> $n scenes in $DEST/trajectory_txt ($(du -sh "$DEST/trajectory_txt" | cut -f1)); SiT has 65."
echo "Next (in your G19 folder):  python webots/walk_model/sit_to_trajectron.py $DEST/trajectory_txt"
