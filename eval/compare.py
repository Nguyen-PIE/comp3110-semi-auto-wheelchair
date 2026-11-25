"""Compare results saved by eval/run_episodes.py or webots/evaluate.py.

    python eval/compare.py A.parquet B.parquet            # B against A, trip by trip (paired)
    python eval/compare.py --table data/episodes/*.parquet   # one row per file, to pick between many runs

Paired mode matches trips by (world, crowd, seed): every policy drives the same trips, so the
difference on each trip is what the policy changed, not luck of the draw. For each metric it prints
the mean on A and on B, the mean difference B - A with a 95% bootstrap interval, and the number of
trips. A difference whose interval includes 0 is not evidence of anything. With success near 100%
and 20 trips per world, a gap of one to three trips will always look like this: use more trips
(--episodes 100) before believing a result.

--by picks how the trips are grouped (default: all together), for example --by scenario or --by crowd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# metric, lower is better?  (None: just report)
METRICS = [
    ("success", False), ("collision", True), ("timeout", True), ("time_to_goal_s", True),
    ("delay_ratio", True), ("stops_per_min", True), ("flicker_stops", True), ("unneeded_stops", True),
    ("waiting_near_people_s", True), ("min_speed_near_people", False), ("peak_decel_mps2", True),
    ("hard_brakes", True), ("safety_stops", True), ("weaves_per_min", True), ("intrusion_s", True),
    ("near_misses", True), ("alarm_false_rate", True), ("pred_err_2s_m", True),
]
TABLE_COLUMNS = ["method", "n", "success", "collision", "timeout", "time_to_goal_s", "delay_ratio",
                 "stops_per_min", "flicker_stops", "unneeded_stops", "min_speed_near_people",
                 "peak_decel_mps2", "hard_brakes", "safety_stops", "weaves_per_min", "pred_err_2s_m"]
RESAMPLES = 5000


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    df["_source"] = p.stem
    if "crowd" not in df.columns:
        df["crowd"] = "-"
    return df


def paired(a: pd.DataFrame, b: pd.DataFrame, by: str | None) -> None:
    key = ["scenario", "crowd", "seed"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    if merged.empty:
        sys.exit("No trips in common: the files must be on the same worlds, crowds and seeds.")
    dropped = len(a) - len(merged), len(b) - len(merged)
    print(f"A = {a['_source'].iloc[0]}   B = {b['_source'].iloc[0]}   "
          f"{len(merged)} trips in common (not matched: {dropped[0]} in A, {dropped[1]} in B)")
    rng = np.random.default_rng(0)
    groups = [("all trips", merged)] if by is None else list(merged.groupby(by, sort=False))
    for name, part in groups:
        rows = []
        for metric, lower_is_better in METRICS:
            if f"{metric}_a" not in part.columns:
                continue
            pair = part[[f"{metric}_a", f"{metric}_b"]].astype(float).dropna()
            if pair.empty:
                continue
            d = (pair[f"{metric}_b"] - pair[f"{metric}_a"]).to_numpy()
            boots = rng.choice(d, size=(RESAMPLES, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            verdict = ""
            if lo > 0 or hi < 0:
                better = (d.mean() < 0) == lower_is_better
                verdict = "B better" if better else "B worse"
            rows.append({"metric": metric, "A": pair.iloc[:, 0].mean(), "B": pair.iloc[:, 1].mean(),
                         "B - A": d.mean(), "95% interval": f"[{lo:+.2f}, {hi:+.2f}]", "n": len(d),
                         "": verdict})
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(f"\n{name}\n" + pd.DataFrame(rows).to_string(index=False))


def table(paths: list[str]) -> None:
    rows = []
    for path in paths:
        df = load(path)
        row = {"method": df["_source"].iloc[0], "n": len(df)}
        for col in TABLE_COLUMNS[2:]:
            if col in df.columns:
                row[col] = pd.to_numeric(df[col], errors="coerce").mean()
        rows.append(row)
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="result files (.parquet or .csv)")
    p.add_argument("--table", action="store_true", help="one row per file instead of a paired comparison")
    p.add_argument("--by", choices=["scenario", "crowd", "world_kind"], help="group the paired comparison")
    args = p.parse_args()
    if args.table:
        table(args.files)
    elif len(args.files) == 2:
        paired(load(args.files[0]), load(args.files[1]), args.by)
    else:
        p.error("give exactly two files to compare, or use --table for any number")


if __name__ == "__main__":
    main()
