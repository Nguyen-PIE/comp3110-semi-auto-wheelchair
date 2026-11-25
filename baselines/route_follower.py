"""A naive driver: follows the route around walls and furniture, ignores people.

Like go_to_goal.py, this is not one of the project's baselines. It gives the
scenarios a sensible automatic driver to watch and override, and a lower bound:
it never hits walls on its own, so every collision it has is with a person.
On floor plans it uses the building plan; in the generated rooms (s1_corridor,
s2_ward, ...) it builds a plan of the room's walls and furniture at each reset.
"""

from __future__ import annotations

import math

import numpy as np

from baselines.go_to_goal import GoToGoal
from envs.floorplan import CLEARANCE_M, FloorPlan, room_plan, route_length

REPLAN_EVERY = 30       # steps (3 s) between checks for people who started or stopped standing still
AVOID_MOVED_M = 0.5     # re-plan if one of them moved this far
WALL_BRAKE_M = 0.15     # slow down when driving on would leave less than this to a wall
SEARCH_AHEAD_M = 4.0    # look for the chair's place on the route this far beyond its current segment
OFF_ROUTE_M = 0.75      # also re-plan if the chair is this far from its route
MAX_DETOUR_M = 4.0      # go around people standing still only if it adds at most this much
SWITCH_SHORTER_M = 3.0  # with keep_open_route: take a new route only if it is this much shorter (or the old is blocked)
TURN_GAIN = 1.5         # turn command per radian of heading error
SLOW_TURN_RAD = 1.0     # past this heading error the chair slows towards turning on the spot ...
SPOT_TURN_RAD = 1.4     # ... which it does past this one (the route doubles back)
UTURN_RAD = 2.1         # with keep_open_route: a new route going this far (120 degrees) from the way the route
                        # goes now sends the chair back the way it came ...
UTURN_WAIT_STEPS = 80   # ... and is taken only once the chair has been held up 8 s waiting for the way on
COMMIT_STEPS = 100      # after turning back, it does not swing round again for 10 s while its new way is open
RECHECK_STEPS = 10      # while it waits, it looks again every second whether the way on has opened


def plan_of(env) -> FloorPlan:
    """The floor plan of a floor-plan scenario, or the plan of the current generated room."""
    u = env.unwrapped
    if hasattr(u, "plan"):
        return u.plan
    return getattr(u, "_room_plan", None) or room_plan(u)


class RouteFollower:
    """Drive along the floor plan's route to the goal (pure pursuit)."""

    def __init__(self, env, lookahead: float = 1.0):
        self.env = env
        self.lookahead = lookahead
        self.slow_radius = GoToGoal().slow_radius
        self.path: list[tuple[float, float]] | None = None   # where the route was planned from, then its waypoints
        self.progress = 0                                    # the path segment the chair is on
        self.steps = 0
        self.avoid: list[tuple[float, float, float]] = []   # circles (x, y, radius) to plan around; none by default
        self.plan: FloorPlan | None = None
        self.fixed: list[tuple[float, float, float]] = []   # those in `avoid` who have stood still for a while
        self.planned_avoid: list[tuple[float, float, float]] = []   # the lists the route was planned with
        self.planned_fixed: list[tuple[float, float, float]] = []
        self.wall_braked = False             # the last steer() was held by a wall right in front
        self.keep_open_route = False         # keep the route while it is open and none is much shorter (see target)
        self.replan_wanted = False           # someone asked for a new route (the assistant: a walker asked to make
                                             # way, someone holding it up); with keep_open_route it is checked first
        self.replan_from_here = False        # plan afresh from where the chair is (after getting out of a gap)
        self.uturn_waiting = 0               # steps held up waiting rather than turning back (0: not waiting)
        self.committed_until = -1            # turned back at this step - COMMIT_STEPS: keep that way until then

    def reset(self) -> None:
        """Call after each env.reset(): a generated room changes at every reset."""
        self.path = None
        self.progress = 0
        self.steps = 0
        self.avoid, self.fixed = [], []
        self.plan = plan_of(self.env)
        self.replan_wanted = self.replan_from_here = False
        self.uturn_waiting, self.committed_until = 0, -1

    @property
    def route(self) -> list[tuple[float, float]] | None:
        """The waypoints still ahead of the chair."""
        return None if self.path is None else self.path[self.progress + 1:]

    def target(self) -> tuple[float, float, float]:
        """The point to drive towards, in the chair's frame (x forward, y left), and the
        straight-line distance left to the goal. Call once per step."""
        u = self.env.unwrapped
        robot = u.robot
        here, goal = (robot.x, robot.y), (robot.goal_x, robot.goal_y)

        if self.plan is None:
            self.plan = plan_of(self.env)
        # Keep the route unless something changed: re-planning from a slightly different spot
        # can give a route bending the other way, and the chair would swing round to follow it.
        # With keep_open_route, people stopping for a moment do not change it either: only someone
        # who has stood on it for a while (in `fixed`), or a new way SWITCH_SHORTER_M shorter.
        # Otherwise, in two-way traffic past a row of benches, it turned back and forth between
        # the two sides as people paused.
        # With keep_open_route, a new route that turns the chair back the way it came is not taken
        # at once either (see _replan_for_people): in a narrow spot, someone who had stood beside
        # the route for 5 s made it turn round on the spot, then turn back once they moved.
        if self.path is None:
            self._replan(here, goal)
        elif not self.keep_open_route:
            if self.replan_wanted or self.replan_from_here or (self.steps % REPLAN_EVERY == 0 and self._avoid_changed()):
                self._replan(here, goal)
            self.replan_wanted = self.replan_from_here = False
        else:
            if self.uturn_waiting and abs(getattr(robot, "vel_x", 0.0)) < 0.1:
                self.uturn_waiting += 1
            if self.replan_wanted or self.replan_from_here or \
                    (self.uturn_waiting and self.steps % RECHECK_STEPS == 0) or \
                    (self.steps % REPLAN_EVERY == 0 and self._avoid_changed()):
                self._replan_for_people(here, goal, fresh=self.replan_from_here)
            self.replan_wanted = self.replan_from_here = False
        target, off_route = self._pursue(here)
        if off_route > OFF_ROUTE_M:
            self._replan(here, goal)
            target, _ = self._pursue(here)
        self.steps += 1

        dx, dy = target[0] - robot.x, target[1] - robot.y
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        return c * dx + s * dy, -s * dx + c * dy, math.dist(here, goal)

    def _replan(self, here, goal) -> None:
        self.path = [tuple(here)] + list(self._plan(self.plan, here, goal))
        self.progress = 0

    def _replan_for_people(self, here, goal, fresh: bool = False) -> None:
        """With keep_open_route: a new route because people moved, stopped, or were asked to make way.
        The route is kept while it is open, unless a new one is SWITCH_SHORTER_M shorter (fresh: the
        new one is taken whenever it goes the same way). A new route that turns the chair back the way
        it came is taken only when the route is blocked and the chair has been held up
        UTURN_WAIT_STEPS waiting (the person may move on), and never in the COMMIT_STEPS after the
        last time it turned back while its route is open (it would swing back as they move)."""
        is_open = self._route_open(here)
        new = [tuple(here)] + list(self._plan(self.plan, here, goal))
        turns_back = self._turns_back(here, new)
        if is_open:
            self.uturn_waiting = 0
            if turns_back and self.steps < self.committed_until:
                return
            shorter = route_length(here, new[1:]) < route_length(here, self.route or [goal]) - SWITCH_SHORTER_M
            if shorter or (fresh and not turns_back):
                self.path, self.progress = new, 0
                if turns_back:
                    self.committed_until = self.steps + COMMIT_STEPS
            return
        if turns_back:
            if self.uturn_waiting < UTURN_WAIT_STEPS:
                self.uturn_waiting = max(self.uturn_waiting, 1)   # wait, and keep looking (see target)
                return
            self.committed_until = self.steps + COMMIT_STEPS
        self.uturn_waiting = 0
        self.path, self.progress = new, 0

    def _heading_along(self, path, here) -> float:
        """The direction (world frame) the chair would drive in from `here` along `path`."""
        kept = self.path, self.progress
        self.path, self.progress = path, (self.progress if path is kept[0] else 0)
        try:
            target, _ = self._pursue(here)
        finally:
            self.path, self.progress = kept
        return math.atan2(target[1] - here[1], target[0] - here[0])

    def _turns_back(self, here, new) -> bool:
        """Whether the route `new` sends the chair back the way it came: it goes UTURN_RAD or more from
        the way its route goes now, and taking it would have the chair turn on the spot (SPOT_TURN_RAD
        or more from the way it faces). After getting out of a gap the chair can already face the new
        way: then taking it does not turn it round."""
        if self.path is None or len(new) < 2:
            return False

        def apart(a, b):
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

        way = self._heading_along(new, here)
        return apart(way, self._heading_along(self.path, here)) >= UTURN_RAD and \
            apart(way, self.env.unwrapped.robot.orientation) >= SPOT_TURN_RAD

    def _pursue(self, here) -> tuple[tuple[float, float], float]:
        """Pure pursuit: the point `lookahead` metres further along the route than the chair, and
        how far the chair is from the route. Progress along the route only moves forward, so
        waypoints the chair has passed are never chased again."""
        path = self.path
        p = np.asarray(here, dtype=float)
        best_d, best_i, best_t = math.inf, self.progress, 0.0
        searched = 0.0
        for i in range(self.progress, len(path) - 1):
            a, b = np.asarray(path[i], float), np.asarray(path[i + 1], float)
            ab = b - a
            length = float(np.linalg.norm(ab))
            t = float(np.clip((p - a) @ ab / max(length * length, 1e-9), 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - p))
            if d < best_d - 1e-9:
                best_d, best_i, best_t = d, i, t
            searched += length
            if searched > SEARCH_AHEAD_M:
                break
        self.progress = best_i
        remaining = self.lookahead * (0.6 + 0.4 * self._roominess())   # follow closely in tight spaces
        for i in range(best_i, len(path) - 1):
            a, b = np.asarray(path[i], float), np.asarray(path[i + 1], float)
            length = float(np.linalg.norm(b - a))
            start = best_t if i == best_i else 0.0
            left = length * (1.0 - start)
            if left >= remaining and length > 0:
                q = a + (start + remaining / length) * (b - a)
                return (float(q[0]), float(q[1])), best_d
            remaining -= left
        return (float(path[-1][0]), float(path[-1][1])), best_d

    def _route_open(self, here) -> bool:
        """Whether the rest of the route keeps the chair's centre CLEARANCE_M clear of everyone in
        `fixed`, as a new route would."""
        points = [tuple(here)] + list(self.route or [])
        for a, b in zip(points[:-1], points[1:]):
            ab = np.subtract(b, a)
            length2 = float(ab @ ab)
            for x, y, r in self.fixed:
                t = 0.0 if length2 < 1e-12 else float(np.clip(np.subtract((x, y), a) @ ab / length2, 0.0, 1.0))
                if math.dist((x, y), np.add(a, t * ab)) < r + CLEARANCE_M:
                    return False
        return True

    def _avoid_changed(self) -> bool:
        """Whether someone started or stopped standing still, or moved, since the route was planned."""
        def differ(old, new):
            if len(old) != len(new):
                return True
            return any(min((math.dist(a[:2], b[:2]) for b in old), default=math.inf) > AVOID_MOVED_M for a in new)
        return differ(self.planned_avoid, self.avoid) or differ(self.planned_fixed, self.fixed)

    def _plan(self, plan, here, goal):
        """The route around everyone standing still if that adds at most MAX_DETOUR_M, else the
        route around those who have stood still for a while (they are not about to move), else
        the plain route, else straight to the goal."""
        self.planned_avoid, self.planned_fixed = list(self.avoid), list(self.fixed)
        base = (plan.route(here, goal, self.fixed) if self.fixed else None) or plan.route(here, goal)
        around = plan.route(here, goal, self.avoid) if self.avoid else None
        if around and (base is None or route_length(here, around) <= route_length(here, base) + MAX_DETOUR_M):
            return around
        return base or around or [goal]

    def steer(self, target_x: float, target_y: float, distance_left: float, speed: float = 1.0) -> np.ndarray:
        """Command towards a target in the chair's frame (x forward, y left), at up to `speed`
        (0 to 1) of full speed. The chair keeps moving while it turns, on the arc through the
        target (pure pursuit), slowing down only as much as its top turn rate needs; it turns on
        the spot only when the target is well behind its side. (Stopping to turn for every bend
        made the chair stop and start whenever its direction changed.) It slows down in tight
        spaces, where it has to follow the route closely."""
        u = self.env.unwrapped
        error = math.atan2(target_y, target_x)
        distance = max(math.hypot(target_x, target_y), 0.3)
        curvature = 2.0 * math.sin(error) / distance               # of the arc through the target
        forward = speed * min(1.0, distance_left / self.slow_radius, 0.5 + 0.5 * self._roominess())
        if abs(curvature) > 1e-6:   # the arc needs turn rate = speed x curvature, and turn rate is limited
            forward = min(forward, u.MAX_ROTATION / (u.MAX_ADVANCE_ROBOT * abs(curvature)))
        forward *= float(np.clip((SPOT_TURN_RAD - abs(error)) / (SPOT_TURN_RAD - SLOW_TURN_RAD), 0.0, 1.0))
        room = self._room_ahead()
        self.wall_braked = room < 0.05 and speed > 0.1   # held by a wall right in front, though asked to go
        forward = max(forward * room, 0.0)
        arc_turn = forward * u.MAX_ADVANCE_ROBOT * curvature / u.MAX_ROTATION
        turn = float(np.clip(max(abs(arc_turn), TURN_GAIN * abs(error) * (1.0 - forward)) * np.sign(error), -1.0, 1.0))
        return np.array([forward, 0.0, turn], dtype=np.float32)

    def _room_ahead(self) -> float:
        """Brake for a wall or piece of furniture just in front: 0 if driving 15 cm forward would
        bring the chair's edge within WALL_BRAKE_M / 5 of it (closer than it is now), 1 if
        there is WALL_BRAKE_M or more."""
        u = self.env.unwrapped
        robot = u.robot
        ahead_x = robot.x + 0.15 * math.cos(robot.orientation)
        ahead_y = robot.y + 0.15 * math.sin(robot.orientation)
        now = self.plan.clearance_at(robot.x, robot.y) - robot.radius
        then = self.plan.clearance_at(ahead_x, ahead_y) - robot.radius
        if then >= WALL_BRAKE_M or then >= now:     # plenty of room, or moving away from it
            return 1.0
        return float(np.clip((then - WALL_BRAKE_M / 5) / (WALL_BRAKE_M * 4 / 5), 0.0, 1.0))

    def _roominess(self) -> float:
        """0 where the chair barely fits (5 cm to spare), 1 with 0.5 m or more to spare."""
        u = self.env.unwrapped
        gap = self.plan.clearance_at(u.robot.x, u.robot.y) - u.robot.radius
        return float(np.clip((gap - 0.05) / 0.45, 0.0, 1.0))

    def act(self, obs: dict) -> np.ndarray:
        return self.steer(*self.target())
