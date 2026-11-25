"""Comfort and prediction metrics for one trip: does the chair drive smoothly around people,
and does it predict where they are going, or does it stop whenever someone walks in front?

eval/metrics.py says whether a trip succeeded. This says how it felt. Record every step with
ComfortRecorder (after env.step), then call summary() at the end of the trip. The numbers go
into the same results rows as eval/metrics.py's, in 2D (eval/run_episodes.py) and in Webots
(webots/evaluate.py). docs/evaluation.md defines every column and says how to read it.

Two kinds of numbers, to be read with care:

* Comfort (stops, speed near people, braking, weaving) is read straight from the chair's own
  speed and turn rate. No assumptions.
* "Hindsight" numbers (unneeded_stops, alarm_false_rate, missed_conflict_s) ask: if the chair had
  carried on at its speed and heading, would it have come within a metre or so of anyone? They use
  where the people really went. But people react to the chair (a stopped chair gets walked past),
  so this is an estimate of what would have happened, not a replay. Compare policies with it;
  do not read one trip's value as truth.

The recorder runs its own people tracker (assist/tracker.py, the same code the assistant uses), so
the prediction numbers are measured for any driver, including route_follower, and in Webots they
see only what the chair can see (line of sight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- definitions (all tunable)
NEAR_PEOPLE_GAP_M = 5.0     # "someone is near": a person within 5 m (edge to edge)
STOP_SPEED = 0.1            # m/s: below this the chair is stopped ...
GO_SPEED = 0.3              # ... and it counts as moving again only above this (so noise does not count as a stop)
FLICKER_S = 1.0             # a stop that lasts under a second is a flicker (stopping and going straight away)
SKIP_START_S = 2.0          # the first seconds (speeding up from standstill) are not judged
COUNTERFACTUAL_S = 3.0      # an unneeded stop: nobody would have come close in the next 3 s had it kept going
UNNEEDED_GAP_M = 0.5        # ... within this gap (the reward's CLOSE_GAP_M)
INTRUSION_GAP_M = 0.5       # time spent moving closer than this to a person (edge to edge)
HARD_DECEL_MPS2 = 1.0       # a "hard brake": slowing faster than this (a starting value; tune with users)
WEAVE_TURN_RAD_S = 0.2      # a turn counts above this rate ...
WEAVE_WINDOW_S = 1.5        # ... and left, right, left again within this time is one weave
ALARM_HORIZON_S = 4.0       # predicted collisions look this far ahead (assist/tracker.py HORIZON_S)
ALARM_CLEAR_GAP_M = 0.5     # an alarm was false if the chair would have stayed this far from the person
ALARM_MISS_GAP_M = 0.3      # a conflict was missed if it would have come closer than this (COLLISION_GAP_M)
PREDICTION_HORIZONS_S = (1.0, 2.0, 3.0)
WALKING_SPEED = 0.3         # prediction error is only judged for people walking faster than this
KEEP_RANGE_M = 12.0         # people further than this from the chair are not recorded
# Proxemic zones (assist/proxemics.py): each step goes to the zone of the nearest person (intimate,
# personal, social, public), or to "nobody" when nobody is within KEEP_RANGE_M. The zones stretch
# forward with the chair's speed, as in the proxemic reward.
ZONE_COLUMNS = ("intimate_s", "personal_s", "social_s", "public_s", "nobody_s")


@dataclass
class Trace:
    """What happened in one trip, one entry per step (after the step)."""
    dt: float
    chair_radius: float
    cruise: float                                   # the chair's top speed (m/s)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    v: list = field(default_factory=list)           # forward speed (m/s)
    w: list = field(default_factory=list)           # turn rate (rad/s)
    safety_stop: list = field(default_factory=list)  # the safety stop overrode the policy this step
    engaged: list = field(default_factory=list)      # the policy (not plain route following) was in charge
    people: list = field(default_factory=list)       # per step: {id: (x, y, radius)}, the true positions
    tracks: list = field(default_factory=list)       # per step: {id: (x, y, vx, vy, alarm)} as the chair estimates

    def __len__(self) -> int:
        return len(self.v)


class ComfortRecorder:
    """Records one trip at a time. Use:  rec = ComfortRecorder(env, driver);  rec.reset() before
    each trip;  rec.update() after each env.step();  rec.summary(...) at the end."""

    def __init__(self, env, driver=None):
        from assist.tracker import PeopleTracker
        self.u = env.unwrapped
        self.driver = driver
        self.tracker = PeopleTracker()
        self.reset()

    def reset(self) -> None:
        self.tracker.reset(0)
        self.trace = Trace(dt=float(self.u.TIMESTEP), chair_radius=float(self.u.ROBOT_RADIUS),
                           cruise=float(self.u.MAX_ADVANCE_ROBOT))

    def update(self) -> None:
        import assist.tracker as at
        u, tr, robot = self.u, self.trace, self.u.robot
        tr.x.append(float(robot.x))
        tr.y.append(float(robot.y))
        tr.heading.append(float(robot.orientation))
        tr.v.append(float(robot.vel_x))
        tr.w.append(float(robot.vel_a))
        controller = getattr(self.driver, "controller", None)
        tr.safety_stop.append(bool(getattr(controller, "safety_stop", False)))
        tr.engaged.append(bool(getattr(controller, "engaged", False)))
        # True positions of everyone (also people the chair cannot see: they are the ground truth).
        everyone = getattr(at, "everyone_in_range", at.all_people)
        tr.people.append({int(p.id): (float(p.x), float(p.y), float(p.width) / 2) for p in everyone(u)
                          if math.dist((p.x, p.y), (robot.x, robot.y)) <= KEEP_RANGE_M})
        tr.tracks.append({int(t.id): (t.x, t.y, t.vx, t.vy, t.collision_predicted)
                          for t in self.tracker.update(u)})

    def summary(self, success: bool = False, time_to_goal_s: float | None = None,
                shortest_route_m: float | None = None) -> dict:
        return compute(self.trace, success, time_to_goal_s, shortest_route_m)


# ---------------------------------------------------------------- the metrics
def _dense(tr: Trace):
    """Per-step, per-person arrays (NaN where a person is not recorded or not tracked)."""
    ids = sorted({i for d in tr.people for i in d} | {i for d in tr.tracks for i in d})
    index = {i: k for k, i in enumerate(ids)}
    T, N = len(tr), len(ids)
    true = np.full((T, N, 2), np.nan)
    est = np.full((T, N, 2), np.nan)
    vel = np.full((T, N, 2), np.nan)
    alarm = np.zeros((T, N), dtype=bool)
    radius = np.full(N, 0.36)
    for t in range(T):
        for i, (x, y, r) in tr.people[t].items():
            true[t, index[i]] = (x, y)
            radius[index[i]] = r
        for i, (x, y, vx, vy, flag) in tr.tracks[t].items():
            est[t, index[i]] = (x, y)
            vel[t, index[i]] = (vx, vy)
            alarm[t, index[i]] = flag
    return true, est, vel, alarm, radius


def _runs(flags: np.ndarray) -> int:
    """Number of separate runs of True."""
    f = np.asarray(flags, dtype=bool)
    return int(np.sum(f & ~np.concatenate(([False], f[:-1])))) if f.size else 0


def stop_events(v: np.ndarray, success: bool) -> list[list]:
    """[[start, end or None], ...]: each time the chair, after moving, slows below STOP_SPEED.
    The standstill at the start and the final stop at the goal are not stops."""
    events: list[list] = []
    moving = False
    for i, s in enumerate(v):
        if not moving and s > GO_SPEED:
            moving = True
            if events and events[-1][1] is None:
                events[-1][1] = i
        elif moving and s < STOP_SPEED:
            moving = False
            events.append([i, None])
    if success and events and events[-1][1] is None:
        events.pop()
    return events


def count_weaves(w: np.ndarray, dt: float) -> int:
    """Left, right, left again (or the reverse) within WEAVE_WINDOW_S; each turn is a stretch of
    turn rate beyond WEAVE_TURN_RAD_S. Counted greedily, so L R L R is one weave."""
    starts, last = [], 0
    for i, rate in enumerate(w):
        sign = 0 if abs(rate) < WEAVE_TURN_RAD_S else (1 if rate > 0 else -1)
        if sign and sign != last:
            starts.append(i)
        if sign:
            last = sign
    window, count, i = WEAVE_WINDOW_S / dt, 0, 0
    while i + 2 < len(starts):
        if starts[i + 2] - starts[i] <= window:
            count += 1
            i += 2
        else:
            i += 1
    return count


def _carry_on_gap(tr: Trace, x, y, heading, speed, k0: int, true: np.ndarray, radius: np.ndarray,
                  steps: int) -> np.ndarray:
    """For a chair that is at (x, y) at step k0 and drives straight on at `speed` along `heading`:
    the smallest edge-to-edge gap to each person over the next `steps` steps. Shape (N,), inf if the
    person is not recorded."""
    T, N = true.shape[0], true.shape[1]
    n = min(steps, T - 1 - k0)
    if n < 0 or N == 0:
        return np.full(N, np.inf)
    k = np.arange(n + 1)
    cx = x + speed * tr.dt * k * math.cos(heading)
    cy = y + speed * tr.dt * k * math.sin(heading)
    p = true[k0:k0 + n + 1]                                   # (n+1, N, 2)
    d = np.hypot(p[:, :, 0] - cx[:, None], p[:, :, 1] - cy[:, None]) - tr.chair_radius - radius[None, :]
    d = np.where(np.isnan(d), np.inf, d)
    return d.min(axis=0)


def compute(tr: Trace, success: bool = False, time_to_goal_s: float | None = None,
            shortest_route_m: float | None = None) -> dict:
    T, dt = len(tr), tr.dt
    nan = float("nan")
    if T < 3:
        return {}
    v, w = np.asarray(tr.v), np.asarray(tr.w)
    x, y, th = np.asarray(tr.x), np.asarray(tr.y), np.asarray(tr.heading)
    true, est, vel, alarm, radius = _dense(tr)
    minutes = T * dt / 60

    # Gap (edge to edge) from the chair to each person, per step.
    gap = np.hypot(true[:, :, 0] - x[:, None], true[:, :, 1] - y[:, None]) - tr.chair_radius - radius[None, :]
    gap = np.where(np.isnan(gap), np.inf, gap)
    nearest = gap.min(axis=1) if gap.shape[1] else np.full(T, np.inf)
    near = nearest < NEAR_PEOPLE_GAP_M
    skip = int(SKIP_START_S / dt)
    judged = np.arange(T) >= skip
    if success:                                    # slowing down to arrive is not hesitation
        judged &= np.arange(T) < T - skip

    # --- stops and starts
    events = stop_events(v, success)
    carry_steps = int(COUNTERFACTUAL_S / dt)
    near_stops = unneeded = resumed = 0
    for start, end in events:
        resumed += end is not None
        near_stops += bool(near[start])
        v_pre = float(v[max(0, start - 10):start + 1].max())
        braking = np.nonzero(v[max(0, start - 10):start + 1] >= 0.8 * v_pre)[0]
        b = max(0, start - 10) + int(braking[-1])   # where the chair began to slow
        if np.min(_carry_on_gap(tr, x[b], y[b], th[b], v_pre, b, true, radius, carry_steps), initial=np.inf) > UNNEEDED_GAP_M:
            unneeded += 1
    waiting = (v < STOP_SPEED) & near & judged

    # --- speed while people are near
    near_judged = near & judged
    cruise = tr.cruise or 1.0
    mean_near = float(v[near_judged].mean() / cruise) if near_judged.any() else nan
    min_near = float(v[near_judged].min() / cruise) if near_judged.any() else nan

    # --- proxemic zones: time in each of the nearest person's zones, entries into the intimate
    # zone, and how much of it the chair drove into (moving, with the person beside or ahead)
    from assist import proxemics as px
    share = np.clip(v / cruise, 0.0, 1.0)
    dx, dy = true[:, :, 0] - x[:, None], true[:, :, 1] - y[:, None]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    zgap = px.zone_gap(lx, ly, radius[None, :], share[:, None]) if gap.shape[1] else np.full((T, 0), np.inf)
    znear = zgap.min(axis=1) if zgap.shape[1] else np.full(T, np.inf)
    zone = np.where(np.isfinite(znear), px.zone_index(znear), 4)          # 4 = nobody within range
    intimate = zone == 0
    ahead_gap = np.where(px.behind(lx), np.inf, zgap) if zgap.shape[1] else zgap
    ahead_near = ahead_gap.min(axis=1) if zgap.shape[1] else np.full(T, np.inf)
    chair_doing = (ahead_near < px.ZONE_EDGES_M[0]) & (v >= STOP_SPEED)
    row_zones = {name: round(float((zone == k).sum() * dt), 1) for k, name in enumerate(ZONE_COLUMNS)}
    row_zones.update({
        "intimate_entries": _runs(intimate),
        "intimate_moving_in_s": round(float(chair_doing.sum() * dt), 1),
        "speed_in_intimate": round(float(v[intimate].mean() / cruise), 3) if intimate.any() else nan,
        "min_zone_gap_m": round(float(znear.min()), 3) if np.isfinite(znear).any() else nan,
        "proxemic_cost": round(float(sum(px.penalty(g, k) for g, k in zip(ahead_near, share))), 3),
    })

    # --- braking and jerk
    a = np.diff(v) / dt
    moving_before = v[:-1] >= STOP_SPEED
    decel = np.where(moving_before, np.maximum(-a, 0.0), 0.0)
    jerk = np.diff(a) / dt if len(a) > 1 else np.zeros(1)
    jerk_valid = v[:-2] >= STOP_SPEED if len(a) > 1 else np.zeros(1, dtype=bool)

    # --- personal space: moving closer than INTRUSION_GAP_M, and how fast the two were closing
    intrusion = (nearest < INTRUSION_GAP_M) & (v >= STOP_SPEED)
    closest_rel_speed = nan
    if np.isfinite(nearest).any() and nearest.min() < NEAR_PEOPLE_GAP_M:
        t = int(np.argmin(nearest))
        n = int(np.argmin(gap[t]))
        t0, t1 = max(0, t - 1), min(T - 1, t + 1)
        pv = (true[t1, n] - true[t0, n]) / ((t1 - t0) * dt)
        cv = np.array([v[t] * math.cos(th[t]), v[t] * math.sin(th[t])])
        if not np.isnan(pv).any():
            closest_rel_speed = float(np.hypot(*(pv - cv)))

    # --- prediction quality: the tracker's straight-line prediction against where people went
    row_pred = {}
    true_speed = np.full(true.shape[:2], np.nan)
    true_speed[:-1] = np.hypot(*(np.diff(true, axis=0) / dt).transpose(2, 0, 1))
    walking = true_speed > WALKING_SPEED
    for horizon in PREDICTION_HORIZONS_S:
        h = int(round(horizon / dt))
        if h >= T:
            row_pred[f"pred_err_{horizon:g}s_m"] = nan
            continue
        guess = est[:-h] + vel[:-h] * horizon          # where the chair expects each person to be
        err = np.hypot(*(guess - true[h:]).transpose(2, 0, 1))
        ok = walking[:-h] & ~np.isnan(err)
        row_pred[f"pred_err_{horizon:g}s_m"] = float(err[ok].mean()) if ok.any() else nan

    # --- alarms: while the chair was moving, would carrying on really have come close?
    false_alarms = alarms = 0
    missed_steps = 0
    horizon_steps = int(ALARM_HORIZON_S / dt)
    for t in np.nonzero(v >= GO_SPEED)[0]:
        seen = ~np.isnan(est[t, :, 0])
        if not seen.any():
            continue
        carry = _carry_on_gap(tr, x[t], y[t], th[t], v[t], t, true, radius, horizon_steps)
        flagged = alarm[t] & seen
        alarms += int(flagged.sum())
        false_alarms += int((flagged & (carry > ALARM_CLEAR_GAP_M)).sum())
        missed_steps += int((seen & ~alarm[t] & (carry < ALARM_MISS_GAP_M)).sum())

    delay = nan
    if success and time_to_goal_s and shortest_route_m:
        delay = time_to_goal_s / (shortest_route_m / cruise)

    return {
        **row_zones,
        "stops": len(events),
        "stops_per_min": round(len(events) / minutes, 2),
        "stops_near_people": near_stops,
        "stops_clear_of_people": len(events) - near_stops,
        "stop_go_cycles": resumed,
        "flicker_stops": sum(end is not None and (end - start) * dt < FLICKER_S for start, end in events),
        "unneeded_stops": unneeded,
        "waiting_near_people_s": round(float(waiting.sum() * dt), 1),
        "mean_speed_near_people": round(mean_near, 3),
        "min_speed_near_people": round(min_near, 3),
        "peak_decel_mps2": round(float(decel.max()), 2) if decel.size else 0.0,
        "hard_brakes": _runs(decel > HARD_DECEL_MPS2),
        "peak_jerk_mps3": round(float(np.abs(jerk[jerk_valid]).max()), 1) if jerk_valid.any() else 0.0,
        "intrusion_s": round(float(intrusion.sum() * dt), 1),
        "closest_rel_speed_mps": round(closest_rel_speed, 2),
        "weaves_per_min": round(count_weaves(w, dt) / minutes, 2),
        "safety_stops": _runs(np.asarray(tr.safety_stop)),
        "policy_active_share": round(float(np.mean(tr.engaged)), 3),
        "delay_ratio": round(delay, 2),
        **{k: round(val, 3) for k, val in row_pred.items()},
        "alarms": alarms,
        "alarm_false_rate": round(false_alarms / alarms, 3) if alarms else nan,
        "missed_conflict_s": round(missed_steps * dt, 1),
    }


# ---------------------------------------------------------------- reporting (shared by the eval scripts)
def comfort_trace_rows(tr: Trace, run_id: str, method: str, scenario: str, seed: int, **extra) -> list[dict]:
    """One row per step (chair state only), for --save-traces: to plot speed profiles or look at a trip."""
    return [{"run_id": run_id, "method": method, "scenario": scenario, "seed": seed, **extra, "step": i,
             "x": tr.x[i], "y": tr.y[i], "heading": tr.heading[i], "speed": tr.v[i], "turn_rate": tr.w[i],
             "safety_stop": tr.safety_stop[i], "policy_active": tr.engaged[i],
             "people_within_5m": sum(math.dist((px, py), (tr.x[i], tr.y[i])) < NEAR_PEOPLE_GAP_M + tr.chair_radius + r
                                     for px, py, r in tr.people[i].values())}
            for i in range(len(tr))]



COMFORT_COLUMNS = dict(
    stops_per_min=("stops_per_min", "mean"),
    stop_go_near=("stops_near_people", "mean"),
    flicker_stops=("flicker_stops", "mean"),
    unneeded_stops=("unneeded_stops", "mean"),
    waiting_s=("waiting_near_people_s", "mean"),
    min_speed_near=("min_speed_near_people", "mean"),
    peak_decel=("peak_decel_mps2", "mean"),
    hard_brakes=("hard_brakes", "mean"),
    weaves_per_min=("weaves_per_min", "mean"),
    safety_stops=("safety_stops", "mean"),
    delay_ratio=("delay_ratio", "mean"),
    alarm_false=("alarm_false_rate", "mean"),
    pred_err_2s=("pred_err_2s_m", "mean"),
)


def print_comfort_table(df, by, title="COMFORT (per trip, averaged)") -> None:
    """Print the comfort columns of a results DataFrame, one row per value of `by`."""
    import pandas as pd
    if "stops_per_min" not in df.columns:
        return
    table = df.groupby(by, sort=False).agg(**COMFORT_COLUMNS)
    with pd.option_context("display.width", 220, "display.precision", 2, "display.max_columns", 30):
        print(f"\n{title}\n" + table.to_string())
