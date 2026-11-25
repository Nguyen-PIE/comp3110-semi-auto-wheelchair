#!/usr/bin/env bash
# Set up an Ubuntu 24.04 (or 22.04) cloud machine with no screen and no GPU, such as the
# AWS c6a.4xlarge, to run and train the wheelchair in Webots.
#
#   git clone <your repo> G19 && cd G19
#   bash webots/cloud/setup_ubuntu.sh
#
# What it does:
#   1. installs the system tools: a virtual screen (Xvfb, so Webots can draw the lidar's view
#      without a monitor), software OpenGL (Mesa, which draws on the CPU instead of a GPU),
#      and the build tools setup.sh needs;
#   2. installs Webots from Cyberbotics' official .deb package;
#   3. runs the project's own setup.sh (Python environment, SocNavGym, floor plans);
#   4. builds all the Webots worlds and drives 2 trips headless as a check.
# Safe to run again.

set -euo pipefail
cd "$(dirname "$0")/../.."
WEBOTS_VERSION="${WEBOTS_VERSION:-R2025a}"
DEB="webots_${WEBOTS_VERSION#R}_amd64.deb"

echo "==> System packages"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  xvfb mesa-utils libgl1 libegl1 libglu1-mesa wget git build-essential python3-venv python3-dev

if [[ ! -x /usr/local/webots/webots ]]; then
  echo "==> Installing Webots $WEBOTS_VERSION"
  tmp="$(mktemp -d)"
  wget -q --show-progress -O "$tmp/$DEB" \
    "https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/${DEB}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$tmp/$DEB"
  rm -rf "$tmp"
fi
echo "Webots installed in /usr/local/webots"

echo "==> Project Python environment (setup.sh)"
bash setup.sh

# shellcheck disable=SC1091
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate wheelchair; \
else source .venv/bin/activate; fi

echo "==> Webots worlds (8 buildings, 3 furnished copies each, plus the test copies)"
python webots/make_world.py all --variants 3

echo "==> Check: 2 trips in the corridor, no screen"
python webots/run.py corridor --episodes 2 --headless

cat <<EOF

All set. To train (inside tmux, so it keeps going if you disconnect):
    tmux new -s train
    $(command -v conda >/dev/null 2>&1 && echo "conda activate wheelchair" || echo "source .venv/bin/activate")
    python webots/train.py --envs 14 --timesteps 300000
Detach with Ctrl-b then d; come back with  tmux attach -t train
EOF
