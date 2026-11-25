#!/usr/bin/env bash
# One-time setup for the 2D simulator on macOS or Linux.
# Usage:  bash setup.sh
#         PYTHON=python3.12 bash setup.sh   (without conda: choose the Python to use)
# Creates a conda environment called "wheelchair" (or a .venv folder if conda
# is not installed), builds Python-RVO2 and SocNavGym from source, installs the
# rest, and runs scripts/check_install.py at the end. Safe to run again.

set -euo pipefail
cd "$(dirname "$0")"
# Keep a copy of everything printed, so problems can be shared and debugged.
exec > >(tee setup.log) 2>&1
echo "setup.sh started $(date)"

ENV_NAME="wheelchair"
PYTHON_VERSION="3.11"
# Exact versions we tested. Update together, and re-run check_install.py.
SOCNAVGYM_URL="https://github.com/gnns4hri/SocNavGym.git"
SOCNAVGYM_COMMIT="95fbc9cebe4357201145a7d0a28c7736c0efe6cf"
RVO2_URL="https://github.com/sybrenstuvel/Python-RVO2.git"
RVO2_COMMIT="c2c46ba8d59556aa10faf03479293236efea154d"

step() { printf "\n==> %s\n" "$1"; }
trap 'echo; echo "Setup stopped at the step above. The full output is in setup.log."' ERR

# 1. Tools needed: git, and a C++ compiler to build Python-RVO2.
if [[ "$(uname)" == "Darwin" ]] && ! xcode-select -p >/dev/null 2>&1; then
  step "Installing Xcode Command Line Tools (a window will open). Run this script again when it finishes."
  xcode-select --install || true
  exit 1
fi
APT_HINT="On Ubuntu or Debian, run:  sudo apt update && sudo apt install -y build-essential git python3-venv python3-dev"
if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed. $APT_HINT"
  exit 1
fi
if [[ "$(uname)" != "Darwin" ]] && ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  echo "No C++ compiler found. $APT_HINT"
  exit 1
fi

# 2. Python environment: conda if available, otherwise a .venv folder.
step "Setting up Python"
if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
    if [[ -x "$base/bin/conda" ]]; then export PATH="$base/bin:$PATH"; break; fi
  done
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  # On Apple Silicon, make sure the environment is native arm64 even if conda
  # itself was installed for Intel Macs.
  if [[ "$(uname)" == "Darwin" && "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    export CONDA_SUBDIR=osx-arm64
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    # conda-forge avoids the Anaconda terms-of-service prompt on the default channels.
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=${PYTHON_VERSION}"
  fi
  conda activate "$ENV_NAME"
  ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
  # No conda: use a Python version we have tested (3.11 first), or the one in $PYTHON.
  PY=""
  for candidate in ${PYTHON:-} python3.11 python3.12 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$(command -v "$candidate")"; break; fi
  done
  if [[ -z "$PY" ]]; then
    echo "Neither conda nor Python 3.10 to 3.12 was found. Install Miniconda (https://docs.conda.io/projects/miniconda) and run this again."
    exit 1
  fi
  if ! "$PY" -c "import ensurepip, venv" >/dev/null 2>&1; then
    echo "$PY cannot create virtual environments. $APT_HINT"
    exit 1
  fi
  if ! "$PY" -c "import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))" >/dev/null 2>&1; then
    echo "The Python development headers for $PY are missing. $APT_HINT"
    exit 1
  fi
  [[ -d .venv ]] || "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ACTIVATE_HINT="source .venv/bin/activate"
fi
python -m pip install --quiet --upgrade pip setuptools wheel cython cmake
ARCH="$(python -c 'import platform; print(platform.machine())')"
echo "Using $(python --version) ($ARCH) at $(command -v python)"

# 3. Make sure the C++ compiler can find its standard headers (macOS).
# CMake 4 no longer passes the macOS SDK to the compiler, and some Command Line
# Tools installs are missing their C++ headers. Both cause
# "fatal error: 'algorithm' file not found", so try each fix in turn.
if [[ "$(uname)" == "Darwin" ]]; then
  step "Checking the C++ compiler"
  echo "macOS $(sw_vers -productVersion), developer tools at $(xcode-select -p)"
  test_dir="$(mktemp -d)"
  printf '#include <algorithm>\n#include <vector>\n#include <cmath>\nint main() { std::vector<int> v{2, 1}; std::sort(v.begin(), v.end()); return (int)std::sqrt(4.0) - 2; }\n' > "$test_dir/t.cpp"
  cxx_works() { c++ -std=c++11 "$test_dir/t.cpp" -o "$test_dir/t" 2>"$test_dir/err.txt" && "$test_dir/t"; }

  default_sdk="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
  sdks=()
  [[ -n "$default_sdk" && -d "$default_sdk" ]] && sdks+=("$default_sdk")
  for s in /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk /Applications/Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.sdk; do
    [[ -d "$s" ]] && sdks+=("$s")
  done
  header_dirs=(/Library/Developer/CommandLineTools/usr/include/c++/v1)

  found=""
  # a) Each macOS SDK, with its own C++ headers or the Command Line Tools' copy.
  for s in ${sdks[@]+"${sdks[@]}"}; do
    export SDKROOT="$s"
    for h in "" "$s/usr/include/c++/v1" "${header_dirs[@]}"; do
      if [[ -n "$h" && ! -f "$h/algorithm" ]]; then continue; fi
      if [[ -n "$h" ]]; then export CPLUS_INCLUDE_PATH="$h"; else unset CPLUS_INCLUDE_PATH; fi
      if cxx_works; then found="SDK $s${h:+ with headers from $h}"; break 2; fi
    done
    echo "  Not working with SDK $s"
  done
  # b) Apple's C++ headers are missing: use conda-forge's copy (libc++ 19, the
  #    version Apple's clang 17 is based on), kept in its own folder.
  if [[ -z "$found" ]] && command -v conda >/dev/null 2>&1 && [[ -n "${sdks[0]:-}" ]]; then
    echo "  Downloading C++ headers from conda-forge"
    headers_prefix="$PWD/third_party/libcxx-headers"
    [[ -d "$headers_prefix" ]] || conda create -y -q -p "$headers_prefix" -c conda-forge --override-channels "libcxx-devel=19"
    export SDKROOT="${sdks[0]}"
    export CPLUS_INCLUDE_PATH="$headers_prefix/include/c++/v1"
    if cxx_works; then found="SDK ${sdks[0]} with conda-forge's C++ headers"; fi
  fi

  if [[ -z "$found" ]]; then
    echo; echo "Compiler error:"; cat "$test_dir/err.txt"
    echo; echo "Diagnostics:"
    ls -la /Library/Developer/CommandLineTools/SDKs/ 2>&1 || true
    pkgutil --pkg-info=com.apple.pkg.CLTools_Executables 2>&1 | head -3 || true
    echo; echo "The C++ compiler cannot find its standard headers. Reinstall the Command Line Tools:"
    echo "    sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install"
    echo "then run this script again."
    exit 1
  fi
  echo "C++ compiler works ($found)."
fi

# 4. Download the two libraries at the exact versions we tested, and the
#    hospital, office and warehouse floor plans from wheelchair-navi.
fetch_commit() {
  local url=$1 dir=$2 commit=$3
  if [[ ! -d "$dir/.git" ]]; then
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
}
step "Downloading Python-RVO2, SocNavGym and the floor plans into third_party/"
fetch_commit "$RVO2_URL" third_party/Python-RVO2 "$RVO2_COMMIT"
fetch_commit "$SOCNAVGYM_URL" third_party/SocNavGym "$SOCNAVGYM_COMMIT"
bash scripts/get_maps.sh

# 5. Build Python-RVO2 (the crowd-avoidance library SocNavGym's people use).
step "Building Python-RVO2"
(
  cd third_party/Python-RVO2
  rm -rf build
  export CMAKE_POLICY_VERSION_MINIMUM=3.5      # its CMakeLists predates CMake 4
  if [[ "$(uname)" == "Darwin" ]]; then
    export CMAKE_OSX_ARCHITECTURES="$ARCH"     # build for the same chip type as Python
    export ARCHFLAGS="-arch $ARCH"
  fi
  python -m pip install --quiet --no-build-isolation .
)

# 6. SocNavGym and everything else.
step "Installing SocNavGym and the other packages"
python -m pip install --quiet ./third_party/SocNavGym
python -m pip install --quiet -r requirements.txt
# Trajectron++, for the 3D simulator's learned walkers (webots/learned_walk.py, --walkers learned)
bash webots/walk_model/get_trajectron.sh
# The speech model for the voice commands (webots/voice.py, always on in run.py), 40 MB, once
bash webots/get_voice_model.sh

# 7. The .env file for the course gateway (voice commands in the rider's own words, webots/interpret.py).
# Made once with the course's address and the model filled in, and the key left empty: put your own in.
# Git never takes it (.gitignore), and a .env that is already there is left as it is.
if [[ -f .env ]]; then
  step "Keeping the .env file that is already there"
else
  step "Making .env for the course gateway: put your own key in it"
  cat > .env <<'ENV'
# The course gateway, for voice commands in the rider's own words (webots/interpret.py).
# Put your own key after COMP3110_API_KEY=. This file stays on your computer: Git never takes it.
COMP3110_MODEL_BASE_URL=https://lllm.cs3110.cse.academy/v1
COMP3110_API_KEY=
COMP3110_MODEL_NAME=gpt-oss-120b-bedrock
ENV
  chmod 600 .env
fi

# 8. Check it all works.
step "Checking the install"
python scripts/check_install.py

cat <<EOF

Setup finished. Each time you open a new terminal:
    cd "$(pwd)"
    ${ACTIVATE_HINT}
    python scripts/demo.py
EOF
