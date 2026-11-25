"""Why does the chair turn on the spot? Drive some trips without a window and find out.

    python webots/spin_check.py fp_house_test --model models/assist_ppo_webots_sit.zip --walkers learned
    python webots/spin_check.py fp_house_test --model models/assist_ppo_webots.zip --walkers learned   # the older model, same trips
    python webots/spin_check.py fp_house_test --model models/assist_ppo_webots_sit.zip --walkers learned --episodes 10 --crowd busy

A step counts as "turning on the spot" when the chair moves forward slower than 0.1 m/s while
turning faster than 0.3 rad/s (about 17 degrees a second). For each such step it records which
part of the code was in charge at that moment:

  escape        the "boxed in for 2 s, look for open space and drive out" routine (assist/controller.py)
  safety stop   the safety stop held the chair (someone right ahead), but the route still pointed sideways
  policy stop   the learned policy was in charge and asked for (almost) no speed, so only the turn was left
  route behind  nobody in the way: the route itself points well behind the chair (a U-turn or a re-plan)
  wall brake    a wall or furniture right in front held the chair
  other         none of the above

It also counts how often the policy flipped its steering from one side to the other while the
chair was stopped (that is what makes it swing left and right). Prints one line per trip and a
summary, and saves every spinning spell (when, where, why, nearest person) to
webots/logs/spin_check.csv so you can replay a seed and watch the worst one.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
sys.path.insert(0, str(ROOT))

from webots.run import VALIDATION_SEEDS  # noqa: E402
from webots import launch  # noqa: E402

SLOW_M_S = 0.1
TURNING_RAD_S = 0.3
CAUSES = ("escape", "safety stop", "policy stop", "route behind", "wall brake", "other")


RECENT_STEPS = 10     # a re-plan this many steps before a spell starts counts as "recent"
SIDE_AHEAD_M = 2.0    # which side of the nearest person the first 2 m of the route passes


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _side_of(points, here, person) -> str:
    """Which side of the chair->person line the route's first SIDE_AHEAD_M passes the person:
    'left' or 'right' (as seen from the chair), with the route's closest approach to them."""
    from assist.controller import _first_metres
    if not points or person is None:
        return ""
    line = _first_metres([tuple(here)] + [tuple(p) for p in points], SIDE_AHEAD_M)
    px, py = person
    best, bq = math.inf, None
    for a, b in zip(line, line[1:]):
        n = max(1, int(math.dist(a, b) / 0.05))
        for k in range(n + 1):
            q = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            d = math.dist(q, (px, py))
            if d < best:
                best, bq = d, q
    if bq is None:
        return ""
    dx, dy = px - here[0], py - here[1]
    cross = dx * (bq[1] - here[1]) - dy * (bq[0] - here[0])
    return f"{'left' if cross > 0 else 'right'} {best:.2f}m"


class RouteWatch:
    """Instance-level shims on the controller's route follower (the classes are not changed): records
    every re-plan (step, why, how far the route direction turned, which side of the nearest person the
    route passes before and after) and who cleared the path when the controller sets it to None."""

    def __init__(self, c, robot):
        self.c, self.robot = c, robot
        self.events: list[dict] = []
        self.step = 0
        self.prev_heading = None      # world-frame route direction at the last target() call
        self.cleared_by = ""
        self.ctx: dict = {}
        base = c.base
        orig_target, orig_replan = base.target, base._replan
        orig_avoid, orig_pursue = base._avoid_changed, base._pursue

        def avoid_changed():
            changed = orig_avoid()
            self.ctx["avoid"] = self.ctx.get("avoid", False) or changed
            return changed

        def pursue(here):
            out = orig_pursue(here)
            self.ctx["pursued"] = self.ctx.get("pursued", 0) + 1
            self.ctx["off"] = out[1]
            return out

        def replan(here, goal):
            if base.path is None:
                reason = f"path cleared ({self.cleared_by})" if self.cleared_by else "start"
            elif self.ctx.get("pursued"):
                reason = f"off route {self.ctx.get('off', 0):.2f}m"
            elif self.ctx.get("avoid"):
                reason = "avoid changed"
            else:
                reason = "other"
            self.ctx.setdefault("reasons", []).append(reason)
            orig_replan(here, goal)

        def target():
            r = self.robot()
            here = (r.x, r.y)
            nearest = min(c.tracks, key=lambda t: math.hypot(t.x - r.x, t.y - r.y), default=None)
            person = None if nearest is None else (nearest.x, nearest.y)
            before_path = base.path
            before = None if before_path is None else list(before_path[base.progress + 1:])
            waiting_before = getattr(base, "uturn_waiting", 0)
            self.ctx = {}
            out = orig_target()
            waiting_after = getattr(base, "uturn_waiting", 0)
            if waiting_after and not waiting_before:
                self.ctx.setdefault("notes", []).append("waiting instead of turning back")
            elif waiting_before and not waiting_after and base.path is before_path:
                self.ctx.setdefault("notes", []).append(f"stopped waiting after {waiting_before / 10:.1f} s, way on open")
            heading = r.orientation + math.atan2(out[1], out[0])
            reasons = self.ctx.get("reasons", [])
            if base.path is not before_path and not reasons:
                reasons = ["new route (people check)"]
            if waiting_before and base.path is not before_path:
                reasons = reasons + [f"after waiting {waiting_before / 10:.1f} s"]
            reasons = reasons + self.ctx.get("notes", [])
            if reasons and reasons != ["start"]:
                self.events.append({
                    "step": self.step, "reason": ", ".join(reasons),
                    "turn_deg": None if self.prev_heading is None else round(math.degrees(_wrap(heading - self.prev_heading))),
                    "side_before": _side_of(before, here, person),
                    "side_after": _side_of(base.route, here, person),
                    "person_id": "" if nearest is None else nearest.id,
                })
            self.prev_heading = heading
            self.cleared_by = ""
            self.step += 1
            return out

        orig_blockers, orig_ask = c._blockers, c._ask

        def blockers(tracks):
            had = base.path is not None
            orig_blockers(tracks)
            if had and base.path is None:
                self.cleared_by = "blocker"

        def ask(here):
            had = base.path is not None
            orig_ask(here)
            if had and base.path is None:
                self.cleared_by = "asked to make way"

        base.target, base._replan, base._avoid_changed, base._pursue = target, replan, avoid_changed, pursue
        c._blockers, c._ask = blockers, ask

    def reset(self) -> None:
        self.events, self.step, self.prev_heading, self.cleared_by = [], 0, None, ""

    def recent(self):
        """The latest re-plan in the last RECENT_STEPS steps, or None."""
        if self.events and self.step - self.events[-1]["step"] <= RECENT_STEPS:
            return self.events[-1]
        return None


def _near_people(c, r, gap_m: float = 0.5) -> str:
    """People within gap_m (edge to edge) of the chair: who the safety stop is holding it for."""
    out = []
    for t in sorted(c.tracks, key=lambda t: t.distance):
        if t.distance < gap_m:
            kind = "standing" if (t.id in c.still or t.speed < 0.2) else "walking"
            out.append(f"{t.id}:{kind} {t.speed:.2f}m/s @{t.distance:.2f}m")
    return "; ".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("world", nargs="?", default="fp_house_test")
    sit_model = ROOT / "models" / "assist_ppo_webots_sit.zip"    # not in Git: the policy in Git if it is missing
    p.add_argument("--model", default=str(sit_model if sit_model.exists() else ROOT / "models" / "assist_ppo_webots.zip"))
    p.add_argument("--walkers", choices=["rules", "learned"], default="learned")
    p.add_argument("--episodes", type=int, default=10, help="trips per crowd kind (default 10)")
    p.add_argument("--crowd", nargs="+", default=["spread", "busy"], choices=["spread", "busy", "events"])
    p.add_argument("--walls", choices=["lidar", "map"], default="lidar")
    p.add_argument("--port", type=int, default=1240)
    p.add_argument("--seeds", type=int, nargs="+", help="these seeds instead of the first --episodes validation seeds")
    p.add_argument("--replans", action="store_true", help="also print every re-plan of each trip")
    p.add_argument("--out", default=str(WEBOTS_DIR / "logs" / "spin_check.csv"), help="where to save the spells")
    args = p.parse_args(argv)

    from assist.controller import FOLLOW_ROUTE, adjustment
    from assist.driver import AssistDriver
    from baselines.route_follower import SLOW_TURN_RAD
    from webots.bridge import WebotsScene
    from webots.run import world_path

    path = world_path(args.world)
    (WEBOTS_DIR / "logs").mkdir(exist_ok=True)
    process = launch.start_webots(path, args.port, fast=True, headless=True,
                                  log=WEBOTS_DIR / "logs" / f"spin_check_{args.port}.log")
    launch.connect_to(args.port)
    spells, totals, trips = [], Counter(), 0
    outcomes, trip_times = Counter(), []
    total_steps = total_flips = 0
    try:
        scene = WebotsScene(path, walls=args.walls, animate=False, walkers=args.walkers)
        driver = AssistDriver(scene, args.model)
        c = driver.controller
        watch = RouteWatch(c, lambda: scene.robot)
        print(f"{args.world}, model {Path(args.model).name}, walkers {args.walkers}", flush=True)
        seeds = args.seeds or list(VALIDATION_SEEDS)[:args.episodes]
        for crowd in args.crowd:
            for seed in seeds:
                scene.crowd_style = crowd
                obs, info = scene.reset(seed=seed)
                driver.reset()
                watch.reset()
                escape_trigger, was_escaping = "", False
                counts, flips, steps, last_side = Counter(), 0, 0, 0
                spell = None
                done = False
                while not done:
                    features = c.observe()
                    engaged = c.needs_adjusting()
                    action = driver.choose(features) if engaged else FOLLOW_ROUTE
                    asked_speed, asked_offset = adjustment(action)
                    fx, fy, _ = c.target_local
                    route_error = abs(math.atan2(fy, fx))
                    stopped_for = _near_people(c, scene.robot)   # before command(): who the stop is for
                    wall_before = getattr(c.base, "wall_braked", False)
                    command = c.command(action)
                    if c.escape_left > 0 and not was_escaping:   # an escape started this step
                        escape_trigger = stopped_for or ("wall brake" if wall_before else "?")
                    was_escaping = c.escape_left > 0
                    obs, _, terminated, truncated, info = scene.step(command)
                    done = terminated or truncated
                    steps += 1
                    r = scene.robot
                    slow = abs(r.vel_x) < SLOW_M_S
                    if slow and engaged and abs(asked_offset) > 0.1:
                        side = 1 if asked_offset > 0 else -1
                        flips += side == -last_side
                        last_side = side
                    if not (slow and abs(r.vel_a) > TURNING_RAD_S):
                        spell = None
                        continue
                    if c.escape_left > 0:
                        cause = "escape"
                    elif c.safety_stop:
                        cause = "safety stop"
                    elif engaged and asked_speed < 0.15:
                        cause = "policy stop"
                    elif route_error > SLOW_TURN_RAD:
                        cause = "route behind"
                    elif getattr(c.base, "wall_braked", False):
                        cause = "wall brake"
                    else:
                        cause = "other"
                    counts[cause] += 1
                    near_t = min(c.tracks, key=lambda t: math.hypot(t.x - r.x, t.y - r.y), default=None)
                    nearest = math.inf if near_t is None else math.hypot(near_t.x - r.x, near_t.y - r.y)
                    if spell is None or spell["cause"] != cause:
                        ev = watch.recent()
                        spell = {"world": args.world, "crowd": crowd, "seed": seed, "start_s": round(info["TIME"], 1),
                                 "seconds": 0.0, "cause": cause, "x": round(r.x, 2), "y": round(r.y, 2),
                                 "nearest_person_m": round(nearest, 2), "turned_deg": 0.0,
                                 "route_error_deg": round(math.degrees(route_error)),
                                 "replanned_recently": "" if ev is None else f"{ev['reason']} ({watch.step - ev['step']} steps ago)",
                                 "route_turn_deg": "" if ev is None or ev["turn_deg"] is None else ev["turn_deg"],
                                 "side_before": "" if ev is None else ev["side_before"],
                                 "side_after": "" if ev is None else ev["side_after"],
                                 "side_person": "" if ev is None else ev["person_id"],
                                 "n_avoid": len(c.base.avoid), "n_fixed": len(c.base.fixed),
                                 "nearest_id": "" if near_t is None else near_t.id,
                                 "nearest_speed": "" if near_t is None else round(near_t.speed, 2),
                                 "nearest_still": "" if near_t is None else near_t.id in c.still,
                                 "escape_trigger": escape_trigger if cause == "escape" else "",
                                 "replans_so_far": "; ".join(f"{e['step'] / 10:.1f}s {e['reason']} turn {e['turn_deg']}"
                                                             for e in watch.events[-3:])}
                        spells.append(spell)
                    spell["seconds"] = round(spell["seconds"] + 0.1, 1)
                    spell["turned_deg"] = round(spell["turned_deg"] + math.degrees(abs(r.vel_a)) * 0.1)
                outcome = "success" if info.get("SUCCESS") else "hit" if info.get("COLLISION") else "timeout"
                spin = sum(counts.values())
                trips += 1
                outcomes[outcome] += 1
                trip_times.append(info["TIME"])
                totals.update(counts)
                total_steps += steps
                total_flips += flips
                why = ", ".join(f"{k} {v / 10:.1f}s" for k, v in counts.most_common()) or "-"
                print(f"  {crowd:<6} seed {seed}: {outcome:<7} {info['TIME']:5.1f} s, on the spot {spin / 10:4.1f} s"
                      f" ({100 * spin / max(1, steps):3.0f}%), side flips {flips:3d}; {why}", flush=True)
                if args.replans:
                    for e in watch.events:
                        print(f"      {e['step'] / 10:5.1f} s  {e['reason']}, route turned {e['turn_deg']} deg, "
                              f"passes person {e['person_id']}: {e['side_before'] or '-'} -> {e['side_after'] or '-'}")
    finally:
        launch.stop_webots(process)

    spin = sum(totals.values())
    print(f"\n{trips} trips: turning on the spot {spin / 10:.0f} s of {total_steps / 10:.0f} s "
          f"({100 * spin / max(1, total_steps):.0f}%), steering side flips while stopped: {total_flips}")
    print(f"  outcomes: {dict(outcomes)}, mean trip {sum(trip_times) / max(1, len(trip_times)):.1f} s")
    for cause in CAUSES:
        if totals[cause]:
            print(f"  {cause:<13} {totals[cause] / 10:6.1f} s  ({100 * totals[cause] / spin:3.0f}%)")
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spells[0]) if spells else ["world"])
        w.writeheader()
        w.writerows(sorted(spells, key=lambda s: -s["seconds"]))
    print(f"Every spell, longest first: {out}")


if __name__ == "__main__":
    main()
    # Webots' controller library ends the process with exit code 1 once the Webots it was attached
    # to has been closed; everything is saved by now, so leave cleanly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
