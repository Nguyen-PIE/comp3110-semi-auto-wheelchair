#!/usr/bin/env bash
# Download the speech model for the voice commands (webots/voice.py, always on in run.py) into
# models/vosk-model-small-en-us-0.15/, once. It is Vosk's small US English model (40 MB to download,
# 68 MB unpacked), so it stays out of Git like the other models:
#     bash webots/get_voice_model.sh

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="vosk-model-small-en-us-0.15"
URL="https://alphacephei.com/vosk/models/$NAME.zip"
DEST="models"

if [[ -f "$DEST/$NAME/am/final.mdl" ]]; then
  echo "The speech model is already in $DEST/$NAME"
  exit 0
fi

echo "Downloading the speech model ($NAME, 40 MB)..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar -o "$tmp/$NAME.zip" "$URL"
mkdir -p "$DEST"
unzip -q "$tmp/$NAME.zip" -d "$DEST"
echo "Speech model in $DEST/$NAME. Check it with: python webots/voice.py --test"
