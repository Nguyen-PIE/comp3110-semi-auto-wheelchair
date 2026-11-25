"""Find Webots, start it, and connect our Python code to it.

Our code runs as an "extern" controller: a normal Python program, started from your
terminal with the project's environment active (so it has PyTorch, Stable-Baselines3
and the rest), that connects to a running Webots. The world says controller "<extern>"
for the chair, so Webots waits for us instead of starting a controller itself.

Webots is looked for in $WEBOTS_HOME, then in its usual install folder:
    macOS    /Applications/Webots.app
    Linux    /usr/local/webots  (the .deb and the tarball), or /snap/webots/current/usr/share/webots
    Windows  C:\\Program Files\\Webots
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

CANDIDATES = {
    "darwin": ["/Applications/Webots.app"],
    "linux": ["/usr/local/webots", "/snap/webots/current/usr/share/webots", "/opt/webots"],
    "win32": [r"C:\Program Files\Webots"],
}


def webots_home() -> Path:
    if os.environ.get("WEBOTS_HOME"):
        return Path(os.environ["WEBOTS_HOME"])
    for candidate in CANDIDATES.get(sys.platform, []):
        if Path(candidate).exists():
            os.environ["WEBOTS_HOME"] = candidate
            return Path(candidate)
    sys.exit("Webots not found. Install it (see webots/README.md) or set WEBOTS_HOME to its folder.")


def add_controller_library_to_path() -> None:
    """Make `from controller import Supervisor` work from any Python."""
    home = webots_home()
    lib = home / ("Contents/lib/controller/python" if sys.platform == "darwin" else "lib/controller/python")
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))


def webots_binary() -> str:
    home = webots_home()
    for candidate in (home / "Contents/MacOS/webots", home / "webots", home / "msys64/mingw64/bin/webots.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("webots") or "webots"


def start_webots(world: str | Path, port: int = 1234, fast: bool = False, headless: bool = False,
                 log: Path | None = None) -> subprocess.Popen:
    """Open `world` in a new Webots, listening on `port`. fast=True runs as fast as the computer
    can and hides the 3D view (for training); headless=True also needs no screen (Linux servers:
    it runs Webots inside a virtual screen, xvfb-run)."""
    cmd = [webots_binary(), f"--port={port}", "--batch", "--stdout", "--stderr"]
    if fast:
        cmd += ["--mode=fast", "--no-rendering", "--minimize"]
    cmd.append(str(Path(world).resolve()))
    if headless and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        if not shutil.which("xvfb-run"):
            sys.exit("No screen and no xvfb-run: install it with  sudo apt install -y xvfb")
        cmd = ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x1024x24"] + cmd
    out = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def connect_to(port: int, robot: str = "wheelchair") -> None:
    """Point the next Supervisor() at the Webots listening on `port`. Call before creating it."""
    os.environ["WEBOTS_CONTROLLER_URL"] = f"ipc://{port}/{robot}"


def stop_webots(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    # Webots was started in its own process group (with xvfb-run, if used): stop all of it.
    kill = (lambda sig: os.killpg(process.pid, sig)) if hasattr(os, "killpg") else \
        (lambda sig: process.terminate() if sig == signal.SIGTERM else process.kill())
    try:
        kill(signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill(getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        pass
    time.sleep(0.2)
