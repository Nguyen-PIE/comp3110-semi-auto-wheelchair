"""How many copies of Webots should one machine run for training? Measure, do not guess.

    python webots/cloud/benchmark.py                          # 4, 8, 12 and 14 copies, 4 minutes each
    python webots/cloud/benchmark.py --envs 6 8 10 --seconds 180
    python webots/cloud/benchmark.py --walls lidar map        # how much does the lidar cost?
    python webots/cloud/benchmark.py --env LP_NUM_THREADS=1   # set a variable for Webots and training (see below)
    python webots/cloud/benchmark.py --dry-run                # only print what it would run

For each setting it starts webots/train.py for a fixed time, reads the steps per second that
train.py prints (policy decisions per second, all copies together, including the learning updates),
and stops it. A c6a.4xlarge has 16 vCPUs but only 8 physical cores, and Webots draws the lidar on the
CPU (Mesa software rendering), so the best number of copies is probably well below 14: the
extra copies fight over the same cores. Pick the smallest number within about 10% of the best.

--env LP_NUM_THREADS=1 is worth one comparison: Mesa's software renderer may start its own threads in
every Webots, which crowds out the others. (Not checked yet; this is how to check.)

It uses ports from --port (default 1400) and writes to a scratch folder, so it never touches
models/ or a Webots you have open on port 1234. Results are also saved to data/benchmarks/.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEPS_PER_S = re.compile(r"\((\d+) steps/s\)")


def memory_used_mb() -> float | None:
    """Memory in use on this machine (Linux only), in MB."""
    try:
        info = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in open("/proc/meminfo"))}
        return (info["MemTotal"] - info["MemAvailable"]) / 1024
    except OSError:
        return None


def run_one(envs: int, walls: str, worlds: list[str], seconds: int, port: int, extra_env: dict) -> dict:
    scratch = tempfile.mkdtemp(prefix="bench_")
    cmd = [sys.executable, str(ROOT / "webots" / "train.py"), "--worlds", *worlds, "--envs", str(envs),
           "--timesteps", "1000000000", "--walls", walls, "--port", str(port),
           "--out", f"{scratch}/bench.zip"]
    lines: list[str] = []
    process = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **extra_env}, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    threading.Thread(target=lambda: [lines.append(l) for l in process.stdout], daemon=True).start()

    base = memory_used_mb()
    peak_mb, loads = 0.0, []
    end = time.time() + seconds
    while time.time() < end and process.poll() is None:
        time.sleep(2)
        used = memory_used_mb()
        if used is not None:
            peak_mb = max(peak_mb, used - (base or 0))
        loads.append(os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0)
    crashed = process.poll() is not None

    # Stop politely first (Ctrl-C: train.py then closes its Webots), then by force.
    if not crashed:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for p in range(port, port + envs):          # any Webots left behind, only on our ports
        subprocess.run(["pkill", "-f", f"webots.*--port={p} "], check=False)
        subprocess.run(["pkill", "-f", f"webots.*--port={p}$"], check=False)
    time.sleep(3)

    rates = [int(m.group(1)) for l in lines for m in [STEPS_PER_S.search(l)] if m]
    return {"envs": envs, "walls": walls, "steps_per_s": rates[-1] if rates else None,
            "per_copy": round(rates[-1] / envs, 1) if rates else None,
            "peak_memory_mb": round(peak_mb) if peak_mb else None,
            "load_avg": round(max(loads), 1) if loads else None,
            "note": "crashed, see log" if crashed and not rates else ("" if rates else "no progress line yet, use --seconds more"),
            "log_tail": "".join(lines[-6:])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, nargs="+", default=[4, 8, 12, 14], help="copies of Webots to try")
    p.add_argument("--walls", nargs="+", choices=["lidar", "map"], default=["lidar"])
    p.add_argument("--worlds", nargs="+", default=["fp_hospital", "fp_office", "fp_depot", "corridor"],
                   help="worlds to drive in, a mix of big and small (default: hospital, office, depot, corridor)")
    p.add_argument("--seconds", type=int, default=240, help="how long to measure each setting (default 240)")
    p.add_argument("--port", type=int, default=1400)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="set an environment variable for the run, e.g. LP_NUM_THREADS=1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    extra_env = dict(e.split("=", 1) for e in args.env)

    combos = list(itertools.product(args.walls, args.envs))
    print(f"{socket.gethostname()}: {os.cpu_count()} vCPUs; {len(combos)} settings x {args.seconds} s "
          f"(about {len(combos) * (args.seconds + 60) // 60} minutes); extra env {extra_env or 'none'}", flush=True)
    if args.dry_run:
        for walls, envs in combos:
            print(f"  would run: train.py --worlds {' '.join(args.worlds)} --envs {envs} --walls {walls}")
        return
    if subprocess.run(["pgrep", "-f", f"webots.*--port={args.port}"], capture_output=True).stdout.strip():
        sys.exit(f"A Webots is already using port {args.port}. Close it or pass --port.")

    results = []
    for walls, envs in combos:
        print(f"-> {envs} copies, walls from {walls} ...", flush=True)
        r = run_one(envs, walls, args.worlds, args.seconds, args.port, extra_env)
        print(f"   {r['steps_per_s']} steps/s ({r['per_copy']} per copy), peak extra memory {r['peak_memory_mb']} MB, "
              f"load {r['load_avg']} {r['note']}", flush=True)
        if r["note"]:
            print("   last output:\n     " + r["log_tail"].replace("\n", "\n     "))
        results.append(r)

    import pandas as pd
    df = pd.DataFrame(results).drop(columns="log_tail")
    df.insert(0, "host", socket.gethostname())
    df["env_vars"] = " ".join(args.env)
    out = ROOT / "data" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}-{socket.gethostname()}.csv"
    df.to_csv(path, index=False)
    print("\n" + df.to_string(index=False))
    done = df.dropna(subset=["steps_per_s"])
    if not done.empty:
        for walls, part in done.groupby("walls"):
            best = part.steps_per_s.max()
            pick = part[part.steps_per_s >= 0.9 * best].envs.min()
            print(f"\nwalls={walls}: best {best} steps/s; the smallest number of copies within 10% of that is {pick}.")
    print(f"\nSaved {path.relative_to(ROOT)}. Paste the table in the group chat so we can compare machines.")


if __name__ == "__main__":
    main()
