"""The crossing test: one person walks across the chair's route, timed to meet it, and the chair has to
deal with it. Every driver gets exactly the same crossings, so the numbers show what the assistant
trained with reinforcement learning does differently from the hand-written rule and from plain route
following (which only follows its route and has no assistant).

    python webots/crossing.py                          # corridor: 20 routes x 3 crossings x 3 drivers
    python webots/crossing.py --world fp_warehouse2 --trips 10
    python webots/crossing.py --headless               # Linux servers without a screen

For each route (one per seed) a person waits beside it 6 m ahead of the chair's start and walks
across at a brisk 1.3 m/s, from the left or from the right, timed to reach the route when the chair
would at full speed; in the third case they walk head-on down the route towards the chair. They do
not give way (a distracted walker, the hardest case): they only stop when the next step would walk
into the chair. Nobody else is in the building. Per driver it prints how often the chair arrived,
collisions split by who closed the gap at the moment of contact: the chair driving into the person
(its speed towards them was higher than theirs towards it), or the person walking into the chair, the smallest gap to the person (edge to edge), how often that
gap was under 0.3 m, hard brakes (slowing faster than 1 m/s^2, as eval/comfort.py) and stops per
crossing, and the time to the goal.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import launch  # noqa: E402

START_AHEAD_M = 6.0        # where the person crosses: this far along the route from the chair's start
SIDE_M = 3.0               # they wait this far to the side of it (less where a wall is closer)
HEAD_ON_M = 6.0            # head-on: they start this much further along the route
WALK_SPEED = 1.3           # m/s
CHAIR_SPEED = 1.2          # m/s, the chair's top speed, to time the crossing
CHAIR_SPEED_UP_S = 0.8     # about how long the chair takes to reach top speed
TIME_LIMIT_S = 40.0
HARD_DECEL_MPS2 = 1.0      # as eval/comfort.py
STOP_SPEED, GO_SPEED = 0.1, 0.3
CLOSE_GAP_M = 0.3
CASES = ("from the left", "from the right", "head-on")
NAMES = {"assist": "learned assistant (RL)", "rule": "rule-based assistant", "route_follower": "route following only"}


def point_along(route, s: float):
    """The point `s` metres along a polyline, and the unit direction there."""
    for a, b in zip(route, route[1:]):
        length = math.dist(a, b)
        if length > 1e-9 and s <= length:
            t = s / length
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        s -= length
    a, b = route[-2], route[-1]
    length = max(math.dist(a, b), 1e-9)
    return tuple(b), ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def route_length(route) -> float:
    return sum(math.dist(a, b) for a, b in zip(route, route[1:]))


def free_spot(plan, centre, direction, wanted: float, line_of_sight):
    """The furthest point up to `wanted` metres from `centre` along `direction` that a person can stand
    on and that can see `centre`, or None."""
    d = wanted
    while d >= 1.0:
        spot = (centre[0] + direction[0] * d, centre[1] + direction[1] * d)
        if plan.clearance_at(*spot) >= 0.45 and line_of_sight(plan, spot, centre):
            return spot
        d -= 0.25
    return None


def crossing_plan(scene, case: str, line_of_sight):
    """Where the person starts, the points they walk through, and when they set off. None if this
    route has no room for the case."""
    route = [tuple(p) for p in scene.route]
    total = route_length(route)
    if total < 4.0:
        return None
    ahead = min(START_AHEAD_M, 0.5 * total)
    centre, (tx, ty) = point_along(route, ahead)
    if case == "head-on":
        start, _ = point_along(route, min(ahead + HEAD_ON_M, total - 0.5))
        if math.dist(start, centre) < 2.0:
            return None
        return start, [centre, tuple(route[0])], 0.0
    side = 1.0 if case == "from the left" else -1.0
    normal = (-ty * side, tx * side)
    start = free_spot(scene.plan, centre, normal, SIDE_M, line_of_sight)
    across = free_spot(scene.plan, centre, (-normal[0], -normal[1]), SIDE_M, line_of_sight)
    if start is None:
        return None
    meet_s = ahead / CHAIR_SPEED + CHAIR_SPEED_UP_S
    set_off = max(0.0, meet_s - math.dist(start, centre) / WALK_SPEED)
    return start, [centre] + ([across] if across else []), set_off


def only_this_person(scene, start, towards):
    """Clear the building of everyone but one person, standing at `start`, walked by this script
    (as a player in players.py would walk them)."""
    crowd = scene.crowd
    people = [p for p in crowd.everyone if p.kind == "person"]
    if not people:
        return None
    p = people[0]
    crowd.walking, crowd.standing = [p], []
    scene.goals.clear()
    scene._walkers.clear()
    p.type, p.player, p._player_move = "dynamic", 1, (0.0, 0.0)
    p._transit, p._stream, p._exit, p._traits = False, None, None, None
    p.goal_x = p.goal_y = None
    p.x, p.y = start
    p.orientation = math.atan2(towards[1] - start[1], towards[0] - start[0])
    return p


def run_crossing(scene, driver, seed: int, case: str, line_of_sight):
    obs, info = scene.reset(seed=seed)
    driver.reset()
    plan = crossing_plan(scene, case, line_of_sight)
    if plan is None:
        return None
    start, waypoints, set_off = plan
    person = only_this_person(scene, start, waypoints[0])
    if person is None:
        return None
    scene.time_limit = TIME_LIMIT_S
    chair = scene.robot
    min_gap, hard_brakes, stops, braking, moving, prev_v, t = math.inf, 0, 0, False, False, 0.0, 0.0
    chair_closing = person_closing = 0.0
    done = False
    while not done:
        before = (chair.x, chair.y, person.x, person.y)
        if t >= set_off and waypoints:            # walk on through the waypoints, then stand
            target = waypoints[0]
            if math.dist((person.x, person.y), target) < 0.3:
                waypoints.pop(0)
            if waypoints:
                target = waypoints[0]
                person.orientation = math.atan2(target[1] - person.y, target[0] - person.x)
                person._player_move = (WALK_SPEED, 0.0)
        if not waypoints:
            person._player_move = (0.0, 0.0)
        obs, _, terminated, truncated, info = scene.step(driver.act(obs))
        done = terminated or truncated
        dt = scene.step_ms / 1000
        t += dt
        gap = math.dist((chair.x, chair.y), (person.x, person.y)) - chair.radius - person.width / 2
        dx, dy = person.x - chair.x, person.y - chair.y          # how fast each closed the gap this step
        n = math.hypot(dx, dy) or 1.0
        chair_closing = ((chair.x - before[0]) * dx + (chair.y - before[1]) * dy) / n / dt
        person_closing = -((person.x - before[2]) * dx + (person.y - before[3]) * dy) / n / dt
        min_gap = min(min_gap, gap)
        v = chair.vel_x
        decel = (prev_v - v) / (scene.step_ms / 1000)
        hard_brakes += decel > HARD_DECEL_MPS2 and not braking
        braking = decel > HARD_DECEL_MPS2
        if v > GO_SPEED:
            moving = True
        elif v < STOP_SPEED and moving:
            stops += 1
            moving = False
        prev_v = v
    hit = bool(info["COLLISION_HUMAN"])
    return dict(seed=seed, case=case, success=bool(info["SUCCESS"]), collision=hit,
                hit_moving=hit and chair_closing > max(person_closing, STOP_SPEED), min_gap=min_gap, hard_brakes=hard_brakes, stops=stops,
                time=info["TIME"] if info["SUCCESS"] else None)


def summarise(name: str, rows: list[dict]) -> str:
    n = len(rows)
    times = [r["time"] for r in rows if r["time"] is not None]
    gaps = sorted(r["min_gap"] for r in rows)
    moving = sum(r['hit_moving'] for r in rows)
    return (f"{NAMES.get(name, name):24s} {n:4d}  {sum(r['success'] for r in rows) / n:7.0%}  "
            f"{moving:9d}  {sum(r['collision'] for r in rows) - moving:9d}  {sum(gaps) / n:8.2f}  {gaps[0]:7.2f}  "
            f"{sum(g < CLOSE_GAP_M for g in gaps) / n:7.0%}  {sum(r['hard_brakes'] for r in rows) / n:8.2f}  "
            f"{sum(r['stops'] for r in rows) / n:6.2f}  {(sum(times) / len(times)) if times else float('nan'):7.1f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default="corridor")
    p.add_argument("--trips", type=int, default=20, help="routes (seeds), each crossed 3 ways (default 20)")
    p.add_argument("--seed", type=int, default=10000, help="first seed (default 10000)")
    p.add_argument("--drivers", nargs="+", default=list(NAMES), help="default: assist rule route_follower")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--headless", action="store_true", help="no screen needed (Linux servers, uses xvfb-run)")
    args = p.parse_args(argv)

    world = ROOT / "webots" / "worlds" / f"{args.world}.wbt"
    process = launch.start_webots(world, args.port, fast=True, headless=args.headless,
                                  log=ROOT / "webots" / "webots.log")
    launch.connect_to(args.port)
    try:
        from baselines.drivers import make_driver
        from webots.bridge import WebotsScene
        from webots.people import line_of_sight
        scene = WebotsScene(world, animate=False, trail=False, stream=False)
        drivers = {name: make_driver(name, scene) for name in args.drivers}
        results = {name: [] for name in args.drivers}
        for seed in range(args.seed, args.seed + args.trips):
            for case in CASES:
                for name, driver in drivers.items():
                    row = run_crossing(scene, driver, seed, case, line_of_sight)
                    if row is not None:
                        results[name].append(row)
                        print(f"{args.world} seed {seed} {case:15s} {NAMES.get(name, name):24s} "
                              f"{'arrived' if row['success'] else 'CHAIR HIT' if row['hit_moving'] else 'walked into' if row['collision'] else 'no arrival':11s} "
                              f"closest {row['min_gap']:5.2f} m  hard brakes {row['hard_brakes']}  stops {row['stops']}",
                              flush=True)
        print(f"\nCrossing test, {args.world}: one person crosses the chair's route, the same crossings for every driver")
        print(f"{'driver':24s} {'runs':>4s}  {'arrived':>7s}  {'chair hit':>9s}  {'walked in':>9s}  {'mean gap':>8s}  {'closest':>7s}  "
              f"{'<0.3 m':>7s}  {'brakes':>8s}  {'stops':>6s}  {'time s':>7s}")
        for name, rows in results.items():
            if rows:
                print(summarise(name, rows))
    finally:
        launch.stop_webots(process)


if __name__ == "__main__":
    main()
