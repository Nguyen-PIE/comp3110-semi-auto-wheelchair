"""The 3D test set: run many trips in Webots without a window and save the results.

    python webots/evaluate.py                                    # the learned 2D policy, 20 trips per test world
    python webots/evaluate.py --model models/assist_ppo_webots.zip   # the policy fine-tuned in 3D
    python webots/evaluate.py --driver route_follower            # any driver from the 2D simulator
    python webots/evaluate.py --worlds fp_hospital fp_office --episodes 10
    python webots/evaluate.py --parallel 4                       # 4 worlds at a time (one Webots each)
    python webots/evaluate.py --crowd spread                     # only the spread-out crowds

The test worlds (webots/splits.py) are things training never sees: a new furniture layout in
each training building (<name>_test) and a building never trained on (fp_house). Make them
with  python webots/make_world.py all --variants 3. Every policy is tested on the same trips:
the validation seeds 10000, 10001, ... in each world, so two runs can be compared trip by trip.
--split test uses the seeds 20000, 20001, ... instead, held back for the final evaluation.

Each trip is driven three times, with three kinds of crowd (--crowd all, the default):
  spread  people anywhere in the building, as on an ordinary day: the main score
  busy    people placed near the chair's route, as in the 2D simulator: a stress test
  events  spread out, plus situations: older people, people stepping out of rooms, ...
The people tracker only sees people in the chair's line of sight (--see-through-walls: as in 2D).

Crashes are split in two: hit_moving (the chair was driving when it touched someone) and
hit_stopped (it stood still for the last half second and someone walked or drove into it).
Both count as collisions.

Results are saved like eval/run_episodes.py's, with the same columns, to
data/episodes/<run_id>.parquet (one row per trip), and the same summary table is printed,
with one row per world and crowd and a total for each kind of world (new furniture, new building).
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing
import sys
import time
from datetime import datetime
from pathlib import Path

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
sys.path.insert(0, str(ROOT))

from webots.run import VALIDATION_SEEDS  # noqa: E402  (10000-10199, as envs/make_env.py)
# Held back for the final evaluation, as envs/make_env.py: never used to train, tune or choose.
TEST_SEEDS = range(20_000, 20_200)
SEEDS = {"validation": VALIDATION_SEEDS, "test": TEST_SEEDS}
from eval.comfort import print_comfort_table  # noqa: E402
from webots import launch, splits  # noqa: E402

# A crash counts as "while stopped" if the chair moved slower than this (m/s) throughout the
# last STOPPED_WINDOW_S before the touch: something walked or drove into a chair standing still.
STOPPED_SPEED = 0.05
STOPPED_WINDOW_S = 0.5


def run_world(world: str, driver_name: str, model: str | None, episodes: int, walls: str,
              port: int, run_id: str, method: str, crowds: tuple[str, ...] = ("spread", "busy", "events"),
              see_through_walls: bool = False, save_traces: bool = False,
              walkers: str = "rules", split: str = "validation") -> tuple[list[dict], list[dict]]:
    """Open `world` in its own Webots on `port`, drive `episodes` trips, return one row each
    (and, with save_traces, one row per step: the chair's speed and turn rate)."""
    from envs.floorplan import route_length
    from eval.comfort import ComfortRecorder, comfort_trace_rows
    from eval.metrics import EpisodeTracker
    from webots.run import world_path

    path = world_path(world)
    log_dir = WEBOTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    process = launch.start_webots(path, port, fast=True, headless=True, log=log_dir / f"evaluate_{port}.log")
    launch.connect_to(port)
    rows, traces = [], []
    try:
        from baselines.drivers import make_driver
        from webots.bridge import WebotsScene
        scene = WebotsScene(path, walls=walls, animate=False, see_through_walls=see_through_walls,
                            walkers=walkers)
        if model:
            from assist.driver import AssistDriver
            driver = AssistDriver(scene, model)
        else:
            driver = make_driver(driver_name, scene)
        trips = [(crowd, seed) for crowd in crowds for seed in list(SEEDS[split])[:episodes]]
        for episode_id, (crowd, seed) in enumerate(trips):
            scene.crowd_style = crowd
            obs, info = scene.reset(seed=seed)
            driver.reset()
            tracker = EpisodeTracker(scene)
            comfort = ComfortRecorder(scene, driver)
            comfort.reset()
            shortest = route_length(scene.route[0], scene.route[1:])
            recent = collections.deque(maxlen=max(1, round(STOPPED_WINDOW_S / tracker.dt)))
            done = False
            while not done:
                obs, _, terminated, truncated, info = scene.step(driver.act(obs))
                tracker.update(info, speed=scene.robot.vel_x, turn_rate=scene.robot.vel_a)
                comfort.update()
                recent.append(abs(scene.robot.vel_x))
                done = terminated or truncated
            summary = tracker.summary()
            stopped = max(recent) < STOPPED_SPEED
            row = {"run_id": run_id, "episode_id": episode_id, "method": method, "scenario": world,
                   "seed": seed, "split": split, "walkers": walkers, **summary,
                   "hit_while_moving": summary["collision"] and not stopped,
                   "hit_while_stopped": summary["collision"] and stopped,
                   "shortest_route_m": round(shortest, 2), "world_kind": splits.kind(world),
                   "simulator": "webots", "walls": walls, "crowd": crowd,
                   **comfort.summary(summary["success"], summary["time_to_goal_s"], shortest)}
            rows.append(row)
            if save_traces:
                traces += comfort_trace_rows(comfort.trace, run_id, method, world, seed, crowd=crowd)
            print(f"  {world} {crowd} seed {seed}: {row['outcome']:<9} {row['time_s']:5.1f} s", flush=True)
    finally:
        launch.stop_webots(process)
    return rows, traces


def main(argv=None) -> None:
    from baselines.drivers import DRIVERS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", nargs="+", default=["test"],
                   help="worlds, or the groups 'test' (default) and 'train' (see webots/splits.py)")
    p.add_argument("--driver", default="auto", choices=DRIVERS,
                   help="auto = the learned assistant (route_follower if there is no trained model)")
    p.add_argument("--model", help="test this policy file with the assist driver, e.g. models/assist_ppo_webots.zip")
    p.add_argument("--episodes", type=int, default=20, help="trips per world (default 20)")
    p.add_argument("--walls", choices=["lidar", "map"], default="lidar")
    p.add_argument("--parallel", type=int, default=1, help="worlds run at the same time, one Webots each")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--crowd", choices=["all", "spread", "busy", "events"], default="all",
                   help="spread: people anywhere (main score); busy: near the route (stress test); "
                        "events: spread out plus older people, people stepping out of rooms, ...")
    p.add_argument("--save-traces", action="store_true",
                   help="also save the chair's speed and turn rate at every step to data/traces/<run_id>.parquet")
    p.add_argument("--see-through-walls", action="store_true",
                   help="let the people tracker see people behind walls, as in 2D")
    p.add_argument("--split", choices=["validation", "test"], default="validation",
                   help="validation: seeds 10000 and up, used while building the rules; test: seeds 20000 "
                        "and up, held back for the final evaluation")
    p.add_argument("--walkers", choices=["rules", "learned"], default="rules",
                   help="how people step: the 2D simulator's rules, or Trajectron++ trained on real "
                        "pedestrians (webots/learned_walk.py)")
    args = p.parse_args(argv)
    crowds = ("spread", "busy", "events") if args.crowd == "all" else (args.crowd,)

    import pandas as pd

    worlds = splits.expand(args.worlds)
    missing = [w for w in worlds if not splits.made(w)]
    if missing:
        sys.exit(f"Not made yet: {', '.join(missing)}\nMake them with:  python webots/make_world.py all --variants 3")
    method = Path(args.model).stem if args.model else args.driver
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-webots-{method}" + ("-test" if args.split == "test" else "")
    print(f"Testing {method} on {len(worlds)} worlds, {args.episodes} trips each with "
          f"{' and '.join(crowds)} crowds, {args.split} seeds from {SEEDS[args.split].start}", flush=True)
    start = time.time()

    jobs = [(w, args.driver, args.model, args.episodes, args.walls, args.port + i, run_id, method, crowds,
             args.see_through_walls, args.save_traces, args.walkers, args.split)
            for i, w in enumerate(worlds)]
    rows, traces = [], []
    if args.parallel > 1:
        # A fresh process for each world: Webots allows one controller connection per process.
        with multiprocessing.get_context("spawn").Pool(args.parallel, maxtasksperchild=1) as pool:
            for world_rows, world_traces in pool.starmap(run_world, jobs, chunksize=1):
                rows += world_rows
                traces += world_traces
    else:
        for job in jobs:
            world_rows, world_traces = run_world(*job)
            rows += world_rows
            traces += world_traces

    df = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:                        # no pyarrow: save a CSV instead
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    if traces:
        traces_dir = ROOT / "data" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(traces).to_parquet(traces_dir / f"{run_id}.parquet", index=False)
        print(f"Saved the speed traces to data/traces/{run_id}.parquet")

    aggregations = dict(
        episodes=("seed", "count"),
        success_rate=("success", "mean"),
        collisions_per_100=("collision", lambda s: 100 * s.mean()),
        hit_moving=("hit_while_moving", "sum"),
        hit_stopped=("hit_while_stopped", "sum"),
        wall_hits_per_100=("collision_wall", lambda s: 100 * s.mean()),
        timeouts=("timeout", "sum"),
        mean_time_to_goal_s=("time_to_goal_s", "mean"),
        near_misses_per_ep=("near_misses", "mean"),
        min_gap_m=("min_gap_to_person_m", "min"),
        mean_jerk=("mean_jerk", "mean"),
        freezes_per_ep=("freezes", "mean"),
    )
    with pd.option_context("display.width", 200, "display.precision", 2):
        for crowd in crowds:
            part = df[df.crowd == crowd]
            per_world = part.groupby("scenario", sort=False).agg(**aggregations)
            per_kind = part.groupby("world_kind", sort=False).agg(**aggregations)
            per_kind.index = [f"ALL {k}" for k in per_kind.index]
            title = {"spread": "SPREAD-OUT CROWDS (main score)", "busy": "BUSY CROWDS (stress test)",
                     "events": "EVENTS (spread-out crowds plus older people, people stepping out of rooms, ...)"}[crowd]
            print(f"\n{title}\n" + pd.concat([per_world, per_kind]).to_string())
            print_comfort_table(pd.concat([part, part.assign(scenario=part.world_kind.map(lambda k: f"ALL {k}"))]),
                                "scenario", title=f"COMFORT, {crowd} crowds (per trip, averaged)")
    print(f"\nSaved {len(df)} trips to {out_path.relative_to(ROOT)} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
