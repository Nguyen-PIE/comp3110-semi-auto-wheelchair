#!/usr/bin/env bash
# Download Trajectron++ (the published walking model, MIT licence) into third_party/ and fix the few
# lines that newer Python, NumPy and PyTorch no longer accept. setup.sh runs this for you.
#     bash webots/walk_model/get_trajectron.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

URL="https://github.com/StanfordASL/Trajectron-plus-plus.git"
COMMIT="1031c7bd1a444273af378c1ec1dcca907ba59830"   # the version we tested
DEST="third_party/Trajectron-plus-plus"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$URL"
fi
if ! git -C "$DEST" cat-file -e "$COMMIT" 2>/dev/null; then
  git -C "$DEST" fetch -q --depth 1 origin "$COMMIT"
fi
git -C "$DEST" checkout -q -f "$COMMIT"
git -C "$DEST" apply "$PWD/webots/walk_model/trajectron_compat.patch"
python -m pip install --quiet dill orjson
echo "Trajectron++ ready in $DEST"
