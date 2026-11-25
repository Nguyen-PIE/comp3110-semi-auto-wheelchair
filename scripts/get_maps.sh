#!/usr/bin/env bash
# Download the hospital, office and warehouse floor plans from wheelchair-navi
# into third_party/floorplans/wheelchair-navi/. setup.sh runs this for you;
# run it on its own if the floor-plan scenarios say the maps are missing:
#     bash scripts/get_maps.sh
# Only the map folder is downloaded (a few hundred KB), not the whole repository.

set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/AutoAILab/wheelchair-navi.git"
COMMIT="2784f9076c428734617b8dcc3445e5c1dae01cc9"   # the version we tested
MAP_DIR="Omniverse/ros_workspace/wheelchair_navigation/map"
DEST="third_party/floorplans/wheelchair-navi"
MAPS=(carter_hospital_navigation carter_office_navigation carter_warehouse_navigation)

missing=0
for m in "${MAPS[@]}"; do
  [[ -f "$DEST/$m.yaml" && -f "$DEST/$m.png" ]] || missing=1
done
if [[ $missing == 0 ]]; then
  echo "Floor plans already in $DEST"
  exit 0
fi

echo "Downloading the floor plans from wheelchair-navi..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" remote add origin "$URL"
git -C "$tmp" sparse-checkout set "$MAP_DIR"
git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$tmp" checkout -q FETCH_HEAD

mkdir -p "$DEST"
for m in "${MAPS[@]}"; do
  cp "$tmp/$MAP_DIR/$m.yaml" "$tmp/$MAP_DIR/$m.png" "$DEST/"
done
echo "Floor plans saved to $DEST"
