#!/usr/bin/env bash
# One-time Webots install for the 3D simulator.
# Usage:  bash webots/install_webots.sh              (from the G19 folder, after bash setup.sh)
#         bash webots/install_webots.sh --reinstall  (replace the Webots already installed)
# Installs Webots R2025a in its usual folder, then checks that the project's Python environment can
# reach it. Safe to run again. On macOS 27 it installs a nightly build instead: R2025a does not start
# there (Webots issue https://github.com/cyberbotics/webots/issues/6976). GitHub keeps only the last
# few nightly builds, so once the one we tested is gone it takes the newest and prints which one.

set -Eeuo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="wheelchair"
RELEASE="R2025a"                    # the worlds are made for Webots R2025a
MAC_NIGHTLY="nightly_2_11_2026"     # the nightly build we tested on macOS 27
DOWNLOADS="https://github.com/cyberbotics/webots/releases/download"
RELEASES_API="https://api.github.com/repos/cyberbotics/webots/releases?per_page=30"

reinstall=0
for arg in "$@"; do
  case "$arg" in
    --reinstall) reinstall=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

mkdir -p webots/logs
# Keep a copy of everything printed, so problems can be shared and debugged.
LOG="webots/logs/install_webots.log"
exec > >(tee "$LOG") 2>&1
echo "install_webots.sh started $(date)"

step() { printf "\n==> %s\n" "$1"; }
trap 'echo; echo "Install stopped at the step above. The full output is in $LOG."' ERR
tmp="$(mktemp -d)"
mnt="$tmp/mount"
cleanup() {
  if [[ -d "$mnt" ]] && command -v hdiutil >/dev/null 2>&1; then hdiutil detach "$mnt" -quiet 2>/dev/null || true; fi
  rm -rf "$tmp"
}
trap cleanup EXIT

download() { curl -L --fail --progress-bar -o "$2" "$1"; }

# 1. The project's Python environment from setup.sh: conda if available, otherwise .venv.
step "Activating the project's Python environment"
if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
    if [[ -x "$base/bin/conda" ]]; then export PATH="$base/bin:$PATH"; break; fi
  done
fi
if command -v conda >/dev/null 2>&1 && conda env list | awk -v env="$ENV_NAME" '$1 == env { found = 1 } END { exit !found }'; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  ACTIVATE_HINT="conda activate ${ENV_NAME}"
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ACTIVATE_HINT="source .venv/bin/activate"
else
  echo "The project's Python environment was not found. Run  bash setup.sh  first, then this script again."
  exit 1
fi
echo "Using $(python --version) from ${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"

# 2. Webots itself, in the folder where webots/launch.py looks for it.
install_on_mac() {
  local v major dmg="$tmp/webots.dmg"
  v="$(sw_vers -productVersion)"
  major="${v%%.*}"
  if [[ -d /Applications/Webots.app && $reinstall == 0 ]]; then
    echo "Webots is already in /Applications. If it does not start, run this again with --reinstall."
    return
  fi
  if (( major >= 27 )); then
    step "Downloading the Webots nightly build $MAC_NIGHTLY (Webots $RELEASE does not start on macOS $v)"
    if ! download "$DOWNLOADS/$MAC_NIGHTLY/webots-$RELEASE.dmg" "$dmg"; then
      # GitHub keeps only the last few nightly builds: take the newest one instead.
      local url
      url="$(curl -fsSL "$RELEASES_API" |
        grep -o '"browser_download_url": *"[^"]*/nightly_[^"]*/webots-R2025a\.dmg"' |
        awk -F'"' 'NR == 1 { print $(NF - 1) }' || true)"
      if [[ -z "$url" ]]; then
        echo "No Webots nightly build found. See https://github.com/cyberbotics/webots/releases"
        exit 1
      fi
      step "$MAC_NIGHTLY is no longer on GitHub. Downloading the newest nightly build instead"
      echo "$url"
      download "$url" "$dmg"
    fi
  else
    step "Downloading Webots $RELEASE"
    download "$DOWNLOADS/$RELEASE/webots-$RELEASE.dmg" "$dmg"
  fi

  step "Installing Webots in /Applications"
  if [[ -d /Applications/Webots.app ]]; then
    local old
    old="$HOME/.Trash/Webots-old-$(date +%Y%m%d-%H%M%S).app"
    while [[ -e "$old" ]]; do old="${old%.app}-1.app"; done
    mv /Applications/Webots.app "$old"
    echo "The Webots that was there is in the Trash: $old"
  fi
  mkdir -p "$mnt"
  hdiutil attach -nobrowse -mountpoint "$mnt" "$dmg"
  ditto "$mnt/Webots.app" /Applications/Webots.app || sudo ditto "$mnt/Webots.app" /Applications/Webots.app
  hdiutil detach "$mnt" -quiet
}

install_on_linux() {
  local deb="webots_${RELEASE#R}_amd64.deb"
  if [[ -x /usr/local/webots/webots && $reinstall == 0 ]]; then
    echo "Webots is already in /usr/local/webots."
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "No apt here. Install Webots $RELEASE by hand: https://github.com/cyberbotics/webots/releases/tag/$RELEASE"
    exit 1
  fi
  step "Downloading Webots $RELEASE"
  download "$DOWNLOADS/$RELEASE/$deb" "$tmp/$deb"
  step "Installing it with apt (asks for your password)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall "$tmp/$deb"
}

step "Installing Webots"
if [[ -n "${WEBOTS_HOME:-}" && -d "$WEBOTS_HOME" && $reinstall == 0 ]]; then
  echo "Using the Webots in WEBOTS_HOME: $WEBOTS_HOME"
else
  case "$(uname)" in
    Darwin) install_on_mac ;;
    Linux) install_on_linux ;;
    *) echo "This script is for macOS and Linux, like setup.sh. Install Webots $RELEASE by hand:"
       echo "https://github.com/cyberbotics/webots/releases/tag/$RELEASE"
       exit 1 ;;
  esac
fi

# 3. Check that Python finds Webots and its controller library (what webots/run.py uses).
step "Checking the install"
python - <<'EOF'
from webots import launch

home = launch.webots_home()
launch.add_controller_library_to_path()
import controller  # noqa: E402,F401  Webots' own Python library

print(f"Webots found in {home}, and Python can use its controller library. All good.")
EOF

cat <<EOF

Webots installed. Each time you open a new terminal:
    cd "$(pwd)"
    ${ACTIVATE_HINT}
    python webots/run.py corridor
EOF
