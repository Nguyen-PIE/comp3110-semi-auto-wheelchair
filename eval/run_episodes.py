"""Run many episodes without a window and save the results.

Examples:
    python eval/run_episodes.py --scenario s1_corridor --episodes 20
    python eval/run_episodes.py --scenario rooms --episodes 50        # the five single-room scenarios
    python eval/run_episodes.py --scenario floorplans --episodes 20   # hospital, office, warehouse
    python eval/run_episodes.py --scenario maps/level2.yaml           # your own floor plan

Results are saved to data/episodes/<run_id>.parquet (one row per episode, with
the length of the path driven and of the shortest route) and a summary table is
printed. --save-paths also saves the chair's position at every step to
data/paths/<run_id>.parquet, to compare the paths of two drivers or versions.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.drivers import DRIVERS, make_driver  # noqa: E402
from envs.make_env import (FLOORPLAN_SCENARIOS, ROOM_SCENARIOS, SCENARIOS, TEST_SEEDS, TRAIN_SEEDS,  # noqa: E402
                           VALIDATION_SEEDS, is_map_path, make_env, scenario_title)
from baselines.route_follower import plan_of  # noqa: E402
from envs.floorplan import route_length  # noqa: E402
from eval.comfort import ComfortRecorder, comfort_trace_rows, print_comfort_table  # noqa: E402
from eval.metrics import EpisodeTracker  # noqa: E402

SPLITS = {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS}
GROUPS = {"all": list(SCENARIOS), "rooms": list(ROOM_SCENARIOS), "floorplans": list(FLOORPLAN_SCENARIOS)}


def shortest_route_m(env) -> float:
    """Length of the shortest route from the chair's start to its goal, around walls and furniture."""
    u = env.unwrapped
    if hasattr(u, "robot_route"):
        return route_length(u.robot_route[0], u.robot_route[1:])
    start, goal = (u.robot.x, u.robot.y), (u.robot.goal_x, u.robot.goal_y)
    return route_length(start, plan_of(env).route(start, goal) or [goal])


def run(scenario: str, driver_name: str, episodes: int, split: str, run_id: str,
        paths: list | None = None, traces: list | None = None, model_path: str | None = None) -> list[dict]:
    """Run the episodes and return one summary row each (eval/metrics.py plus eval/comfort.py). With
    `paths`, also append one row per step with the chair's position, so the paths driven can be
    compared later; with `traces`, one row per step with its speed and turn rate."""
    env = make_env(scenario)
    driver = make_driver(driver_name, env, model_path)
    seeds = list(SPLITS[split])[:episodes]
    rows = []
    for episode_id, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        driver.reset()
        tracker = EpisodeTracker(env)
        comfort = ComfortRecorder(env, driver)
        comfort.reset()
        shortest = shortest_route_m(env)
        robot = env.unwrapped.robot
        steps = [(0, float(robot.x), float(robot.y))]
        done = False
        while not done:
            action = driver.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.update(info, speed=float(env.unwrapped.robot.vel_x),
                           turn_rate=float(env.unwrapped.robot.vel_a))
            comfort.update()
            done = terminated or truncated
            steps.append((len(steps), float(robot.x), float(robot.y)))
        summary = tracker.summary()
        row = {"run_id": run_id, "episode_id": episode_id, "method": driver_name,
               "scenario": scenario, "seed": seed, "split": split, **summary,
               "shortest_route_m": round(shortest, 2),
               **comfort.summary(summary["success"], summary["time_to_goal_s"], shortest)}
        rows.append(row)
        if traces is not None:
            traces += comfort_trace_rows(comfort.trace, run_id, driver_name, scenario, seed)
        if paths is not None:
            paths += [{"run_id": run_id, "method": driver_name, "scenario": scenario, "seed": seed,
                       "step": k, "x": x, "y": y} for k, x, y in steps]
        print(f"  {scenario} seed {seed}: {row['outcome']:<9} {row['time_s']:5.1f} s", flush=True)
    env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="s1_corridor",
                        help=f"one of {', '.join(SCENARIOS)}; a group: {', '.join(GROUPS)}; or a floor-plan .yaml path")
    parser.add_argument("--driver", default="auto", choices=DRIVERS,
                        help="auto = the learned assistant (route_follower if there is no trained model)")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per scenario")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", help="the assist driver's policy file (default models/assist_ppo.zip)")
    parser.add_argument("--name", help="name used in the run id and the method column (default: the driver)")
    parser.add_argument("--save-traces", action="store_true",
                        help="also save the chair's speed and turn rate at every step to data/traces/<run_id>.parquet")
    parser.add_argument("--save-paths", action="store_true",
                        help="also save the chair's position at every step to data/paths/<run_id>.parquet")
    args = parser.parse_args()

    if args.split == "test":
        sys.exit("The test seeds are reserved for the final evaluation in Week 10.")

    scenarios = GROUPS.get(args.scenario, [args.scenario])
    for scenario in scenarios:
        if scenario not in SCENARIOS and not is_map_path(scenario):
            parser.error(f"unknown scenario '{scenario}'")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.name or args.driver}"
    start = time.time()
    rows = []
    paths = [] if args.save_paths else None
    traces = [] if args.save_traces else None
    for scenario in scenarios:
        print(f"Running {args.episodes} episodes of {scenario_title(scenario)} with {args.driver}...")
        try:
            rows += run(scenario, args.driver, args.episodes, args.split, run_id, paths, traces, args.model)
        except (FileNotFoundError, ValueError) as exc:  # map not downloaded, or driver does not fit
            print(f"  Skipped: {exc}")
    if not rows:
        sys.exit("No episodes were run.")

    df = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"
    df.to_parquet(out_path, index=False)
    if paths:
        paths_dir = ROOT / "data" / "paths"
        paths_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(paths).to_parquet(paths_dir / f"{run_id}.parquet", index=False)
        print(f"Saved the paths driven to data/paths/{run_id}.parquet")

    if traces:
        traces_dir = ROOT / "data" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(traces).to_parquet(traces_dir / f"{run_id}.parquet", index=False)
        print(f"Saved the speed traces to data/traces/{run_id}.parquet")

    summary = df.groupby("scenario", sort=False).agg(
        episodes=("seed", "count"),
        success_rate=("success", "mean"),
        collisions_per_100=("collision", lambda s: 100 * s.mean()),
        wall_hits_per_100=("collision_wall", lambda s: 100 * s.mean()),
        timeouts=("timeout", "sum"),
        mean_time_to_goal_s=("time_to_goal_s", "mean"),
        near_misses_per_ep=("near_misses", "mean"),
        min_gap_m=("min_gap_to_person_m", "min"),
        mean_jerk=("mean_jerk", "mean"),
        freezes_per_ep=("freezes", "mean"),
    )
    with pd.option_context("display.width", 200, "display.precision", 2):
        print("\n" + summary.to_string())
    print_comfort_table(df, "scenario")
    print(f"\nSaved {len(df)} episodes to {out_path.relative_to(ROOT)} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main()
