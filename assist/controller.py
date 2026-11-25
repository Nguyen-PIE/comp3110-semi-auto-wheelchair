"""Route following, adjusted by a learned policy: the core of the shared-control assistant.

Each step:
  1. the route follower gives the direction along the route to the goal (the
     route goes around people standing still, and is only re-planned when
     something changes);
  2. the people tracker estimates where everyone nearby is walking, and how close
     they will come to the chair;
  3. the wall sensor measures the distance to walls and furniture around the chair;
  4. if someone is on the next 3 m of the route or predicted to come too close
     (or was, in the last second), the policy (trained with reinforcement
     learning, see training/train_assist.py) looks at all of that and chooses
     how much to slow down and how far to steer left or right of the route.
     Otherwise the chair follows its route at full speed;
  5. that adjusted direction and speed become the chair's command. Like a real
     powered chair, it brakes at once but speeds up gently (from stopped to
     full speed in about a second). The steering offset swerves quickly but
     comes back to the route gently, and is cut back near walls on that side;
  6. a safety stop overrides the policy: if someone is ahead of the chair and
     closer than 0.4 m (less for someone off to the side), or brushing past its
     side, it does not move forward (it can still turn).
     This is the first piece of the planned safety layer; the policy learns to
     slow down and steer early enough that it rarely triggers;
  7. if the safety stop has held the chair for 2 s (boxed in between people, for
     example), it looks all round for open space, using the wall sensor and the
     people it tracks, turns towards the most open direction (nearest the route
     to the goal if several are open) and drives out of the gap at low speed. The
     route follower then re-plans from wherever the chair ends up.

The route follower re-plans from where the chair is when it is off its route, so
after the user takes control and hands it back, the chair carries on to the same
goal.
"""

from __future__ import annotations

import math

import numpy as np

from assist.sensors import MAX_RANGE_M, N_RAYS, wall_ranges
from assist.tracker import HORIZON_S, SENSOR_RANGE_M, PeopleTracker
from baselines.route_follower import WALL_BRAKE_M, RouteFollower

MAX_STEER_RAD = 0.6      # the policy can steer up to about 35 degrees off the route
MAX_SPEEDUP_PER_STEP = 0.1   # the speed factor rises by at most this much per 0.1 s step
MAX_STEER_CHANGE = 0.1   # rad per step: the steering offset moves at most this much (1 rad/s)
SAFETY_GAP_M = 0.4       # the safety stop: no driving forward with someone this close straight ahead
BRUSHING_GAP_M = 0.1     # ... or with someone this close beside it
CLOSING_GAP_M = 0.25     # ... and someone walking in towards its front half this close: they may curve into it
PASSING_GAP_M = 0.03     # driving past someone standing still needs at least this much room
STILL_SPEED = 0.2        # m/s: people slower than this are planned around as obstacles ...
MOVING_SPEED = 0.4       # ... until they walk faster than this (so the route does not flip back and forth)
FIXED_STEPS = 50         # people standing still for 5 s are planned around however long the detour
REMEMBER_STEPS = 600     # someone standing in the way is still planned around for 60 s after the chair loses
                         # sight of them (driving off round them, it would forget them and turn back) ...
CLEARED_M = 3.0          # ... unless the chair, this close to their spot and able to see it, finds nobody
GONE_RADIUS_M = 1.0      # within this of it, or sees them this far from it (they walked off)
WALL_STOP_M = 0.1        # no steering offset towards a wall this close to the chair's edge ...
WALL_FREE_M = 0.5        # ... the full offset beyond this
PATH_AHEAD_M = 3.0       # the policy only acts for people on the next 3 m of the route ...
PATH_MARGIN_M = 0.25     # ... (closer than this to the strip the chair drives through) or predicted too close
ENGAGE_HOLD_STEPS = 10   # and keeps acting for 1 s after that
STEER_BACK_PER_STEP = 0.04   # rad per step: the steering offset returns to the route gently (0.4 rad/s)
STUCK_STEPS = 20         # stopped by the safety stop for 2 s: look for a way out
ESCAPE_ROOM_M = 0.7      # a way out needs this much clear space (chair's edge to the first obstacle)
ESCAPE_ABORT_M = 0.3     # ... and is given up if the space ahead shrinks below this
ESCAPE_SPEED = 0.35      # speed factor while getting out
ESCAPE_DRIVE_M = 1.0     # drive this far, then hand back to the route follower
ESCAPE_MAX_STEPS = 60    # or for at most 6 s
ESCAPE_TURN_COST = 0.5   # preferring the way to the goal: this many metres of room per half turn away from it
ESCAPE_ON_TURN_COST = 1.5   # ... and, among ways within a quarter turn of it, this many (keep close to the route:
                            # a way out off to the side left the route behind the chair, and it turned round)
ESCAPE_ANGLES = 24       # directions tried, evenly spread all round
PERSON_MARGIN_M = 0.02   # room kept between the chair and a person it drives past
WALKER_MARGIN_M = 0.1    # ... and a person walking: their next step and the tracker's noise
LOOK_AHEAD_S = (0.1, 0.2, 0.3, 0.5)   # moments ahead at which nobody walking may come that close
ASK_GAP_M = 0.6          # a walker this near right in front (edge to edge) holding the chair up is asked to make way ...
ASK_AHEAD_COS = 0.7      # ... (within 45 degrees of its way) ...
ASK_HELD_STEPS = 30      # ... once it has been held up 3 s (kept within STALLED_M of one spot) ...
ASK_EVERY_STEPS = 30     # ... and again after 3 s
ASK_LANE_M = 5.0         # they make way for this much of the chair's way ahead
ASKED_STEPS = 60         # someone stepping aside when asked is not planned round for 6 s (they are moving off)
STALLED_M = 0.3          # (held up: not more than this from where it was, however it edges about)
BLOCKER_STEPS = 40       # held up 4 s by someone right in front (even someone shuffling about): plan round them
BLOCKER_GAP_M = 0.6      # ... someone this near (edge to edge) ...
BLOCKER_KEEP_STEPS = 100 # ... for 10 s, unless they walk off
CROWDED_GAP_M = 0.5      # someone walking this close (edge to edge), not behind it: the chair goes ...
CROWDED_SPEED = 0.5      # ... at half speed at most, since a walker who turns can be 15-20 cm from where
                         # the tracker last put them, and at full speed that is gone in 0.2 s
OVERTAKEN_GAP_M = 0.3    # someone walking faster right beside the chair (this near) ...
OVERTAKEN_SPEED = 0.15   # ... is let past: the chair slows to this share of full speed
N_PEOPLE = 4             # the nearest people the policy looks at
PERSON_FEATURES = 8
OBS_SIZE = 2 + 1 + 2 + 2 + N_RAYS + N_PEOPLE * PERSON_FEATURES
FOLLOW_ROUTE = np.array([1.0, 0.0], dtype=np.float32)   # the action for "full speed, on the route"


def _first_metres(points, metres: float) -> list:
    """The polyline `points` cut off after `metres`."""
    out, left = [points[0]], metres
    for a, b in zip(points, points[1:]):
        d = math.dist(a, b)
        if d >= left:
            out.append((a[0] + (b[0] - a[0]) * left / max(d, 1e-9), a[1] + (b[1] - a[1]) * left / max(d, 1e-9)))
            return out
        out.append(b)
        left -= d
    return out


def adjustment(action) -> tuple[float, float]:
    """Policy output in [-1, 1]^2 -> (speed factor 0 to 1, steering offset in radians, + is left)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float((a[0] + 1.0) / 2.0), float(a[1] * MAX_STEER_RAD)


class AssistController:
    def __init__(self, env):
        self.env = env
        self.base = RouteFollower(env)
        self.base.keep_open_route = True      # people pausing do not turn it back and forth between two ways
        self.tracker = PeopleTracker()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.speed = 0.0                      # speed factor actually applied (after the gentle speed-up)
        self.offset = 0.0                     # steering offset actually applied (rad, after smoothing)
        self.still: set[int] = set()          # ids of people currently treated as standing still
        self.still_steps: dict[int, int] = {}  # for how many steps each of them has stood still
        self.remembered: dict[int, tuple] = {}   # standing people out of sight: id -> (x, y, radius, still steps, last seen)
        self.sides: dict[int, tuple[int, int]] = {}   # the rule's chosen side per person: id -> (side, step)
        self.steps = 0
        self.engage_until = -1                # the policy is used up to this step
        self.safety_stop = False              # whether the safety stop overrode the policy this step
        self.engaged = False                  # whether the policy's adjustment was used this step
        self.asking = False                   # asking a walker in the way to make way (this step)
        self.asked_at = -ASK_EVERY_STEPS
        self.held_steps = 0                   # steps in a row the chair has been (nearly) still
        self.stalled_at, self.stalled_steps = None, 0   # where it has stayed within STALLED_M of, and how long
        self.blockers: dict[int, tuple] = {}  # people who held the chair up: id -> (x, y, radius, since step)
        self.made_way: dict[int, int] = {}    # people asked to make way who are stepping aside: id -> step asked
        self.blocked_steps = 0                # steps in a row the safety stop has held the chair still
        self.escape_left = 0                  # steps left of driving out of a gap (0: not escaping)
        self.escape_heading = 0.0             # the way out (world frame, radians)
        self.escape_from = (0.0, 0.0)
        self.features = np.zeros(OBS_SIZE, dtype=np.float32)
        self.target_local = (1.0, 0.0, 0.0)   # route direction in the chair's frame, and distance left
        self.ranges = np.full(N_RAYS, MAX_RANGE_M, dtype=np.float32)

    @property
    def route(self):
        return self.base.route

    @property
    def tracks(self):
        return self.tracker.tracks

    def reset(self, seed: int | None = None) -> None:
        self.base.reset()
        self.tracker.reset(seed)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.speed = 0.0
        self.offset = 0.0
        self.still = set()
        self.still_steps = {}
        self.remembered = {}
        self.sides = {}
        self.steps = 0
        self.engage_until = -1
        self.safety_stop = False
        self.engaged = False
        self.blocked_steps = self.escape_left = 0
        self.asking, self.asked_at, self.held_steps = False, -ASK_EVERY_STEPS, 0
        self.stalled_at, self.stalled_steps = None, 0
        self.blockers, self.made_way = {}, {}

    def _ahead(self, t) -> float:
        """How far round from the way the route goes a person is: 1 straight ahead, 0 beside, -1 behind."""
        robot = self.env.unwrapped.robot
        fx, fy, _ = self.target_local
        way = robot.orientation + math.atan2(fy, fx)
        return math.cos(math.atan2(t.y - robot.y, t.x - robot.x) - way)

    def _blockers(self, tracks) -> None:
        """Someone right in front who has held the chair up for BLOCKER_STEPS is planned round from
        now on, even if they shuffle about (a walker who is not looking, or waiting for the chair to
        move first): the chair finds another way instead of both waiting for good. Not someone it
        has just asked to make way, who is stepping aside (it would turn round as they go)."""
        seen = {t.id: t for t in tracks}
        if self.held_steps >= BLOCKER_STEPS:
            for t in tracks:
                if t.distance < BLOCKER_GAP_M and self._ahead(t) > 0.3 and t.id not in self.blockers \
                        and self.steps - self.made_way.get(t.id, -ASKED_STEPS) >= ASKED_STEPS:
                    self.blockers[t.id] = (t.x, t.y, t.radius + 0.1, self.steps)
                    self.base.replan_wanted = True   # plan again, round them (not straight back the way it came)
        for key, (x, y, r, since) in list(self.blockers.items()):
            t = seen.get(key)
            if self.steps - since > BLOCKER_KEEP_STEPS or (t is not None and math.dist((t.x, t.y), (x, y)) > 1.5):
                del self.blockers[key]
            elif t is not None:
                self.blockers[key] = (t.x, t.y, r, since)
        self.base.fixed = self.base.fixed + [(x, y, r) for x, y, r, _ in self.blockers.values()]

    def _crowded(self) -> bool:
        """Whether someone walking is within CROWDED_GAP_M of the chair and not behind it."""
        robot = self.env.unwrapped.robot
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        for t in self.tracks:
            if t.distance >= CROWDED_GAP_M or t.id in self.still or t.speed < STILL_SPEED:
                continue
            px, py = t.x - robot.x, t.y - robot.y
            if (c * px + s * py) / max(math.hypot(px, py), 1e-6) > -0.3:
                return True
        return False

    def _overtaken(self) -> bool:
        """Whether someone walking faster than the chair, the same way, is right beside it: the chair
        lets them get ahead rather than drive along next to someone who may step across it."""
        robot = self.env.unwrapped.robot
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        for t in self.tracks:
            if t.id in self.still or t.distance > OVERTAKEN_GAP_M:
                continue
            px, py = t.x - robot.x, t.y - robot.y
            beside = abs((c * px + s * py) / max(math.hypot(px, py), 1e-6)) < 0.6
            along = t.vx * c + t.vy * s
            if beside and along > robot.vel_x + 0.3:
                return True
        return False

    def _ask(self, here) -> None:
        """A walker holds the chair up right in front of it (in a doorway or a narrow corridor, or
        stopped in its way): held up 3 s, it asks them to make way, as a visitor says "Excuse me",
        and they step aside if there is room (bridge.py make_way), wait for it to pass and walk on.
        People standing to talk or wait are never asked: the route goes round them."""
        u, robot = self.env.unwrapped, self.env.unwrapped.robot
        fx, fy, _ = self.target_local
        way = robot.orientation + math.atan2(fy, fx)          # the way the route goes from here
        near = [t for t in self.tracks if t.distance <= ASK_GAP_M
                and math.cos(math.atan2(t.y - robot.y, t.x - robot.x) - way) > ASK_AHEAD_COS and u.askable(t.x, t.y)]
        self.asking = bool(near) and self.stalled_steps >= ASK_HELD_STEPS   # (stopping and starting counts too)
        if not self.asking or self.steps - self.asked_at < ASK_EVERY_STEPS:
            return
        self.asked_at = self.steps
        goal = (robot.goal_x, robot.goal_y)
        lane = [here] + list(self.base.plan.route(here, goal) or [goal])   # its way, people aside
        for t in near:
            if u.make_way(t.x, t.y, _first_metres(lane, ASK_LANE_M)):
                self.made_way[t.id] = self.steps
        self.base.replan_wanted = True        # plan again from what the chair sees now (see RouteFollower.target)

    def _remember(self, standing, tracks, here) -> list[tuple]:
        """Keep planning round people who stood in the way after they drop out of sight. Without this,
        a chair taking the long way round two people chatting in a passage forgets them once it can
        no longer see them, plans the short way again, comes back, sees them, and so on. Returns the
        remembered people who are not in `standing`."""
        for t in standing:   # someone seen again keeps the time they were already known to stand still
            still = max(self.still_steps[t.id], self.remembered.get(t.id, (0, 0, 0, 0, 0))[3])
            self.still_steps[t.id] = still
            self.remembered[t.id] = (t.x, t.y, t.radius + 0.1, still, self.steps)
        current = {t.id for t in standing}
        seen = {t.id: t for t in tracks}
        for key, (x, y, _r, _still, last_seen) in list(self.remembered.items()):
            if key in current:
                continue
            walked_off = key in seen and math.dist((seen[key].x, seen[key].y), (x, y)) > GONE_RADIUS_M
            gone = (math.dist(here, (x, y)) < CLEARED_M and self._can_see(here, (x, y))
                    and not any(math.dist((t.x, t.y), (x, y)) < GONE_RADIUS_M for t in tracks))
            if walked_off or gone or self.steps - last_seen > REMEMBER_STEPS:
                del self.remembered[key]
        return [v for key, v in self.remembered.items() if key not in current]

    def _can_see(self, a, b, step_m: float = 0.2) -> bool:
        """Whether nothing on the floor plan (a wall, or a table the chair's sensors cannot see past)
        stands between a and b."""
        plan = self.base.plan
        if plan is None:
            return True
        n = max(1, int(math.dist(a, b) / step_m))
        return all(plan.clearance_at(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) > 0.02
                   for i in range(1, n))

    def observe(self) -> np.ndarray:
        """Sense the scene and return the policy's input vector. Call once per step."""
        u = self.env.unwrapped
        robot = u.robot
        fx, fy, dist_left = self.base.target()
        self.target_local = (fx, fy, dist_left)
        tracks = self.tracker.update(u)
        self.steps += 1
        # People standing still are planned around, like any other obstacle.
        self.still = {t.id for t in tracks
                      if t.speed < STILL_SPEED or (t.id in self.still and t.speed < MOVING_SPEED)}
        self.still_steps = {i: self.still_steps.get(i, 0) + 1 for i in self.still}
        standing = [t for t in tracks if t.id in self.still and t.distance < 6.0]
        kept = self._remember(standing, tracks, (robot.x, robot.y))
        self.base.avoid = [(t.x, t.y, t.radius + 0.1) for t in standing] + [(x, y, r) for x, y, r, _, _ in kept]
        self.base.fixed = [(t.x, t.y, t.radius + 0.1) for t in standing if self.still_steps[t.id] >= FIXED_STEPS] + \
            [(x, y, r) for x, y, r, still, _ in kept if still >= FIXED_STEPS]
        self.held_steps = self.held_steps + 1 if abs(robot.vel_x) < 0.1 else 0
        if self.stalled_at is None or math.dist((robot.x, robot.y), self.stalled_at) > STALLED_M:
            self.stalled_at, self.stalled_steps = (robot.x, robot.y), 0
        else:
            self.stalled_steps += 1
        self._blockers(tracks)
        self.asking = False
        if getattr(u, "askable", None) is not None:
            self._ask((robot.x, robot.y))
        self.ranges = wall_ranges(self.base.plan, robot.x, robot.y, robot.orientation)
        if any(t.collision_predicted or self.on_route_ahead(t) for t in tracks):
            self.engage_until = self.steps + ENGAGE_HOLD_STEPS

        norm = math.hypot(fx, fy) or 1.0
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        cvx, cvy = robot.vel_x * c, robot.vel_x * s
        features = [fx / norm, fy / norm, min(dist_left, 10.0) / 10.0,
                    robot.vel_x / 1.2, robot.vel_a / 0.8,
                    float(self.last_action[0]), float(self.last_action[1])]
        features += list(self.ranges / MAX_RANGE_M)
        for i in range(N_PEOPLE):
            if i < len(tracks):
                t = tracks[i]
                px, py = t.x - robot.x, t.y - robot.y
                rvx, rvy = t.vx - cvx, t.vy - cvy
                features += [(c * px + s * py) / SENSOR_RANGE_M, (-s * px + c * py) / SENSOR_RANGE_M,
                             (c * rvx + s * rvy) / 2.0, (-s * rvx + c * rvy) / 2.0,
                             float(np.clip(t.distance, 0.0, 2.0)) / 2.0,
                             float(np.clip(t.gap_closest, -1.0, 4.0)) / 4.0, t.t_closest / HORIZON_S, 1.0]
            else:
                features += [0.0] * (PERSON_FEATURES - 1) + [0.0]
        self.features = np.asarray(features, dtype=np.float32)
        return self.features

    def observe_cached(self) -> np.ndarray:
        """The input vector from the last observe(), without sensing again."""
        return self.features

    def someone_in_the_way(self) -> bool:
        """Whether driving forward would bring the chair too close to someone: within SAFETY_GAP_M
        of a person straight ahead, less for a person off to the side (driving forward closes the
        gap more slowly), and within BRUSHING_GAP_M of someone brushing past its side. Someone walking
        in towards its front half within CLOSING_GAP_M is waited for too: walking past, they can curve
        into its side as it sets off (the tracker follows them a step behind)."""
        robot = self.env.unwrapped.robot
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        for t in self.tracks:
            px, py = t.x - robot.x, t.y - robot.y
            towards = (c * px + s * py) / max(math.hypot(px, py), 1e-6)   # 1 straight ahead, 0 beside
            if towards > 0 and t.distance < SAFETY_GAP_M * towards:
                return True
            if towards > 0 and t.distance < CLOSING_GAP_M and t.t_closest > 0 and t.id not in self.still \
                    and t.speed >= STILL_SPEED:
                return True
            if t.distance >= BRUSHING_GAP_M or towards <= -0.5:
                continue
            if t.id not in self.still:
                return True        # someone walking past right beside it: let them pass
            # Someone standing still beside it: wait only if driving on would touch them.
            sideways = abs(-s * px + c * py)
            if towards > 0 and sideways - robot.radius - t.radius < PASSING_GAP_M:
                return True
        return False

    def clearance_along(self, heading: float) -> float:
        """How far the chair could drive in the world-frame direction `heading` before its edge meets a
        wall, furniture or a person (up to the wall sensor's range). On the floor plan it stops where the
        route follower's wall brake would hold it (the lidar can see open floor past a shelf corner the
        brake will not pass)."""
        robot = self.env.unwrapped.robot
        rel = (heading - robot.orientation + math.pi) % (2 * math.pi) - math.pi
        k = int(round(rel / (2 * math.pi / N_RAYS)))
        gap = min(float(self.ranges[(k + j) % N_RAYS]) for j in (-1, 0, 1)) - robot.radius
        dx, dy = math.cos(heading), math.sin(heading)
        plan = getattr(self.base, "plan", None)
        if plan is not None:
            now = plan.clearance_at(robot.x, robot.y) - robot.radius
            d = 0.0
            while d < gap:
                d += 0.05
                then = plan.clearance_at(robot.x + d * dx, robot.y + d * dy) - robot.radius
                if then <= WALL_BRAKE_M / 5 and then < now:
                    gap = d - 0.05
                    break
        for t in self.tracks:
            px, py = t.x - robot.x, t.y - robot.y
            along, across = px * dx + py * dy, abs(px * dy - py * dx)
            reach = robot.radius + t.radius + PERSON_MARGIN_M
            if along > 0 and across < reach:
                gap = min(gap, along - math.sqrt(reach ** 2 - across ** 2) - 0.0)
        return gap

    def open_heading(self) -> float | None:
        """The most open direction to drive out of a gap (world frame), preferring the way to the goal:
        any with ESCAPE_ROOM_M of clear space, scored by how much room it has (up to 2 m) less
        ESCAPE_TURN_COST per half turn away from the route. A way within a quarter turn of the route
        beats any way back (backing off only helps when every way on is closed). None if boxed in
        all round."""
        robot = self.env.unwrapped.robot
        fx, fy, _ = self.target_local
        to_route = math.atan2(fy, fx)             # in the chair's frame
        best, best_key = None, (False, -math.inf)
        for k in range(ESCAPE_ANGLES):
            rel = -math.pi + 2 * math.pi * k / ESCAPE_ANGLES
            room = self.clearance_along(robot.orientation + rel)
            if room < ESCAPE_ROOM_M:
                continue
            away = abs((rel - to_route + math.pi) % (2 * math.pi) - math.pi)
            on = away <= math.pi / 2 + 1e-6
            key = (on, min(room, 2.0) - (ESCAPE_ON_TURN_COST if on else ESCAPE_TURN_COST) * away / (math.pi / 2))
            if key > best_key:
                best, best_key = robot.orientation + rel, key
        return best

    def escape_command(self) -> np.ndarray | None:
        """The command for driving out of a gap the safety stop has held the chair in (or a wall corner
        its route runs into), or None when not escaping. Starts after STUCK_STEPS stopped; ends after
        ESCAPE_DRIVE_M or ESCAPE_MAX_STEPS, or when the way out closes."""
        robot = self.env.unwrapped.robot
        if self.escape_left <= 0:
            # held still by someone right in front, or by a wall corner the route runs along
            stopped = (self.safety_stop or getattr(self.base, "wall_braked", False)) and abs(robot.vel_x) < 0.05
            self.blocked_steps = self.blocked_steps + 1 if stopped else 0
            if self.blocked_steps < STUCK_STEPS:
                return None
            heading = self.open_heading()
            if heading is None:
                return None
            self.escape_heading, self.escape_left = heading, ESCAPE_MAX_STEPS
            self.escape_from = (robot.x, robot.y)
            self.blocked_steps = 0
        self.escape_left -= 1
        if math.dist((robot.x, robot.y), self.escape_from) >= ESCAPE_DRIVE_M or \
                self.clearance_along(self.escape_heading) < ESCAPE_ABORT_M:
            self.escape_left = 0
            self.base.replan_from_here = True   # out: plan from here, not chase a point on the route behind
            return None
        if self.escape_left == 0:
            self.base.replan_from_here = True
        self.safety_stop = False
        self.speed = min(ESCAPE_SPEED, self.speed + MAX_SPEEDUP_PER_STEP)
        rel = self.escape_heading - robot.orientation
        command = self.base.steer(math.cos(rel), math.sin(rel), 5.0, self.speed)
        u = self.env.unwrapped
        ahead_m = 2 * 0.1 * max(float(command[0]) * getattr(u, "MAX_ADVANCE_ROBOT", 1.2), robot.vel_x)
        if self.would_touch(ahead_m):   # turn on the spot towards the way out first, never into someone
            command[0] = 0.0
            self.speed = 0.0
        return command

    def would_touch(self, ahead_m: float) -> bool:
        """Whether driving `ahead_m` straight on would bring the chair within PASSING_GAP_M of someone
        and closer than it is now (the safety stop is off while getting out of a gap, so this keeps
        the chair from clipping a person beside it as it turns and drives away). People walking are
        followed where they are going over the next half second (LOOK_AHEAD_S), with
        WALKER_MARGIN_M to spare: one of them walking past is not driven into."""
        robot = self.env.unwrapped.robot
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        reach = max(LOOK_AHEAD_S)
        for t in self.tracks:
            px, py = t.x - robot.x, t.y - robot.y
            now = math.hypot(px, py) - robot.radius - t.radius
            after = math.hypot(px - c * ahead_m, py - s * ahead_m) - robot.radius - t.radius
            if after < PASSING_GAP_M and after < now:
                return True
            if t.id in self.still or t.speed < STILL_SPEED:
                continue
            if now < WALKER_MARGIN_M and after < now - 0.005:   # someone walking right beside it: not an inch closer
                return True
            for dt in LOOK_AHEAD_S:              # the chair covers ahead_m over the whole look ahead
                moved = ahead_m * dt / reach
                qx, qy = px + t.vx * dt - c * moved, py + t.vy * dt - s * moved
                standing = math.hypot(px + t.vx * dt, py + t.vy * dt) - robot.radius - t.radius
                then = math.hypot(qx, qy) - robot.radius - t.radius
                if then < WALKER_MARGIN_M and then < standing:   # closer than if it stayed put
                    return True
        return False

    def on_route_ahead(self, t) -> bool:
        """Whether a person is on the next PATH_AHEAD_M of the route, and not walking away from
        the chair along it."""
        robot = self.env.unwrapped.robot
        if t.t_closest <= 0.0 and t.id not in self.still:      # walking away: no need to wait for them
            return False
        reach = robot.radius + t.radius + PATH_MARGIN_M
        a = np.array([robot.x, robot.y])
        p = np.array([t.x, t.y])
        left = PATH_AHEAD_M
        for waypoint in self.route or []:
            b = np.asarray(waypoint, dtype=float)
            ab = b - a
            length = float(np.linalg.norm(ab))
            if length > left:
                b, length = a + ab * (left / length), left
                ab = b - a
            k = float(np.clip((p - a) @ ab / max(length * length, 1e-9), 0.0, 1.0))
            if np.linalg.norm(a + k * ab - p) < reach:
                return True
            left -= length
            if left <= 0:
                break
            a = b
        return False

    def room_to_steer(self, offset: float) -> float:
        """How much of a steering offset to allow (0 to 1): none towards a wall or piece of
        furniture closer than WALL_STOP_M to the chair's edge on that side, all of it beyond
        WALL_FREE_M. The route keeps the chair clear of walls; this stops the policy steering
        it into one while going round someone."""
        if abs(offset) < 1e-6:
            return 1.0
        quarter = N_RAYS // 4
        rays = range(1, quarter + 1) if offset > 0 else range(3 * quarter, N_RAYS)
        gap = float(min(self.ranges[i] for i in rays)) - self.env.unwrapped.robot.radius
        return float(np.clip((gap - WALL_STOP_M) / (WALL_FREE_M - WALL_STOP_M), 0.0, 1.0))

    def needs_adjusting(self) -> bool:
        """Whether anyone matters right now: someone predicted to come too close, or someone on the
        route just ahead, in the last ENGAGE_HOLD_STEPS. Otherwise the chair simply follows its
        route at full speed. (Holding on for a moment stops the policy switching on and off.)"""
        return self.steps <= self.engage_until

    def command(self, action) -> np.ndarray:
        """Turn the policy's adjustment into the chair's command."""
        self.engaged = self.needs_adjusting()
        if not self.engaged:
            action = FOLLOW_ROUTE
        speed, offset = adjustment(action)
        self.safety_stop = self.someone_in_the_way()
        escape = self.escape_command()
        if escape is not None:
            self.last_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            return escape
        if not self.safety_stop:          # someone walking about to cut right across it: wait for them
            robot = self.env.unwrapped.robot
            ahead_m = 2 * 0.1 * max(min(speed, self.speed + MAX_SPEEDUP_PER_STEP) *
                                    getattr(self.env.unwrapped, "MAX_ADVANCE_ROBOT", 1.2), robot.vel_x)
            self.safety_stop = ahead_m > 0.0 and self.would_touch(ahead_m)
        if self.safety_stop:
            speed = 0.0
        elif self._overtaken():
            speed = min(speed, OVERTAKEN_SPEED)
        elif self._crowded():
            speed = min(speed, CROWDED_SPEED)
        self.speed = min(speed, self.speed + MAX_SPEEDUP_PER_STEP)
        offset *= self.room_to_steer(offset)
        # Swerve as quickly as needed, but come back to the route gently.
        back = abs(offset) < abs(self.offset) and offset * self.offset >= 0   # easing off, not switching sides
        step = STEER_BACK_PER_STEP if back else MAX_STEER_CHANGE
        self.offset = float(np.clip(offset, self.offset - step, self.offset + step))
        self.last_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)   # what was applied
        fx, fy, dist_left = self.target_local
        co, so = math.cos(self.offset), math.sin(self.offset)
        return self.base.steer(fx * co - fy * so, fx * so + fy * co, dist_left, self.speed)
