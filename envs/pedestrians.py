"""How the simulated people walk, in every scenario (rooms and floor plans).

SocNavGym moves people with a social force model: a pull towards their
destination plus pushes away from walls, furniture and each other. On its own
that made people freeze in front of the wheelchair (the pull and the push
cancel out), stop dead whenever they turned, and swap destinations at random
(after 50 s of real time, so it even depended on how fast the computer was).
This module replaces the parts that matter for a wheelchair study:

  - Each walking person has a destination and walks a route to it (A* on the
    floor plan, or on a plan of the room's walls and furniture), planned around
    the chair. They only get a new destination when they arrive. If the chair
    is parked on their destination, being next to it counts as arriving.
  - Passing the chair: a person who sees the chair on their path (taking its
    motion into account) chooses a side once and walks past it with room to
    spare, as people pass a wheelchair in a corridor. Attentive people (70%,
    `prob_to_avoid_robot`) react about 3 s ahead and leave about 0.5 m.
    Distracted people react only about 1 s ahead and leave about 0.15 m, which
    gives the assistant the hard cases it must handle.
  - Where there is no room to pass (a narrow spot), they step to the side and
    wait for the chair, then carry on to the same destination.
  - Speed and direction change smoothly (they accelerate at up to 2 m/s^2 and
    never stop just to turn), each person has their own walking speed, and
    they slow down as they arrive.
  - Each person has a character (envs/behaviours.py): some are on the phone,
    some stop now and then or wait a while on arriving, and some walk in pairs
    or threes. This module applies it: group members aim for their place
    beside the leader, and stopped people stand still (but still step aside
    for the chair).
  - Someone who gets stuck (for example two people blocking each other) plans
    a new route to the same destination after 3 s.

`prepare()` runs after each reset and `before_step()` before each step; both
are hooked into SocNavGym by envs/socnavgym_fixes.py.
"""

from __future__ import annotations

import math

import random

import numpy as np

from envs import behaviours
from envs.floorplan import FloorPlan, room_plan

ATTENTIVE_LOOK_S = 3.0      # attentive people react to the chair this far ahead
DISTRACTED_LOOK_S = 1.0     # distracted people only this far ahead
ATTENTIVE_GAP_M = 0.5       # room attentive people leave when passing the chair (edge to edge)
DISTRACTED_GAP_M = 0.15
ACCELERATION = 2.0          # m/s^2: how quickly people change speed or direction
ROUTE_AROUND_CHAIR_M = 0.3  # routes keep this much extra room around the chair
WAYPOINT_REACHED_M = 1.0    # move on to the next corner of the route within this distance
ARRIVED_NEXT_TO_CHAIR_M = 1.3   # a destination the chair is parked on counts as reached this close
STUCK_S = 3.0               # re-plan the route after this long without getting closer ...
STUCK_MOVE_M = 0.3          # ... by this much along it
YIELD_SPEED = 0.4           # m/s when stepping aside for the chair
REPLAN_GAP_STEPS = 20       # take another way round the chair at most every 2 s
WALL_SLIDE_M = 0.15         # within this of a wall, people walk along it rather than into it
EASE_AWAY_SPEED = 0.3       # m/s: how quickly people too close to the chair ease away from it


def plan_of(env) -> FloorPlan:
    """The floor plan, or the plan of the current room (built in prepare())."""
    return env.plan if hasattr(env, "plan") else env._room_plan


def _rot(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# --------------------------------------------------------------------------- routes
def prepare(env) -> None:
    """After a reset: build the room plan (rooms only) and give everyone a route."""
    if not hasattr(env, "plan"):
        env._room_plan = room_plan(env)
    env._walkers = {}
    for human in env.dynamic_humans:
        human._pass_side = None
        human._replan = False
    rng = random.Random(int(np.random.randint(0, 2**31 - 1)))   # (seeded with the scene, as np.random is)
    behaviours.assign(env, list(env.dynamic_humans), rng, behaviours.settings(getattr(env, "_behaviour", None)),
                      plan_of(env))
    before_step(env)


def standing_people(env) -> list[tuple[float, float, float]]:
    """Circles (x, y, radius) around the people who never move: standing alone or in groups."""
    people = list(env.static_humans)
    for group in list(env.static_interactions):
        people.extend(group.humans)
    people.extend(i.human for i in env.h_l_interactions)
    return [(h.x, h.y, h.width / 2 + 0.1) for h in people]


def chair_can_reach_goal(env) -> bool:
    """Whether the chair starts and finishes clear of walls, furniture and the people who never
    move, and can get from one to the other around them. Scenes where it cannot are drawn again."""
    plan = plan_of(env)
    start, goal = (env.robot.x, env.robot.y), (env.robot.goal_x, env.robot.goal_y)
    if plan.clearance_at(*start) < env.robot.radius + 0.1:   # starts (nearly) touching a wall or plant
        return False
    if not plan.connects(start, goal):
        return False
    standing = standing_people(env)
    for x, y, radius in standing:   # nobody standing on the chair's start or goal
        if min(math.dist((x, y), start), math.dist((x, y), goal)) < radius + env.robot.radius + 0.2:
            return False
    return not standing or plan.route(start, goal, standing) is not None


def _destination(env, human):
    goal = env.goals.get(human.id)
    if goal is None:
        return float(human.goal_x), float(human.goal_y)
    return float(goal.x), float(goal.y)


def _plan_route(env, human, destination, extra=(), margin=ROUTE_AROUND_CHAIR_M):
    plan = plan_of(env)
    chair = [(env.robot.x, env.robot.y, env.robot.radius + margin)]
    here = (human.x, human.y)
    return (plan.route(here, destination, chair + list(extra), for_people=True)
            or plan.route(here, destination, chair, for_people=True)
            or plan.route(here, destination, for_people=True) or [destination])


def _route_left(state, here) -> float:
    """How far a person still has to walk along their route."""
    points = state["waypoints"][state["index"]:]
    return math.dist(here, points[0]) + sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def _everyone(env) -> list:
    people = list(env.dynamic_humans) + list(env.static_humans)
    for group in list(env.static_interactions) + list(env.moving_interactions):
        people.extend(group.humans)
    return people + [i.human for i in env.h_l_interactions]


def _chair_on(env, x, y) -> bool:
    robot = env.robot
    return math.dist((x, y), (robot.x, robot.y)) < robot.radius + env.HUMAN_DIAMETER / 2 + 0.2


def before_step(env) -> None:
    """Point each walking person at the next corner of their route to their destination."""
    walkers = env._walkers
    limit = int(round(STUCK_S / env.TIMESTEP))
    for human in env.dynamic_humans:
        if human.goal_x is None:
            continue
        traits = behaviours.traits_of(human)
        if traits is not None and traits.leader is not None:   # walks with a group: aim for their place
            _aim_at_place(env, human, walkers)
            continue
        destination = _destination(env, human)
        state = walkers.get(human.id)
        if state is None or state["destination"] != destination:   # arrived, and given a new one
            state = {"destination": destination, "waypoints": _plan_route(env, human, destination),
                     "index": 0, "still": 0}
            state["best"] = _route_left(state, (human.x, human.y))
            walkers[human.id] = state
        if getattr(human, "_replan", False):   # the chair blocks the way: another route to the same place
            human._replan = False
            if env.ticks - state.get("replanned", -99) > REPLAN_GAP_STEPS:
                state["waypoints"], state["index"] = _plan_route(env, human, destination, margin=0.6), 0
                state["replanned"] = env.ticks
        # Stuck (no progress along their route for a while, and not waiting for the chair): plan
        # again to the same place, around the chair and whoever is close.
        left = _route_left(state, (human.x, human.y))
        if left < state["best"] - STUCK_MOVE_M:
            state["best"], state["still"] = left, 0
        elif (human._pass_side != 0 and math.dist((human.x, human.y), destination) > ARRIVED_NEXT_TO_CHAIR_M
              and not behaviours.is_stopped(human)):
            state["still"] += 1
            if state["still"] >= limit:
                others = [(o.x, o.y, o.width / 2 + 0.2) for o in _everyone(env)
                          if o is not human and math.dist((o.x, o.y), (human.x, human.y)) < 2.5]
                state["waypoints"], state["index"] = _plan_route(env, human, destination, others), 0
                state["best"], state["still"] = _route_left(state, (human.x, human.y)), 0
        waypoints = state["waypoints"]
        here = np.array([human.x, human.y])
        while state["index"] < len(waypoints) - 1:
            point = np.asarray(waypoints[state["index"]])
            following = np.asarray(waypoints[state["index"] + 1])
            past = (here - point) @ (following - point) > 0          # already beyond this corner
            if np.linalg.norm(here - point) < WAYPOINT_REACHED_M or past or _chair_on(env, *point):
                state["index"] += 1
            else:
                break
        target = waypoints[state["index"]]
        human._destination = destination
        human._destination_blocked = _chair_on(env, *destination)
        if (human.goal_x, human.goal_y) != tuple(target):
            human.goal_x, human.goal_y = float(target[0]), float(target[1])


def _aim_at_place(env, human, walkers) -> None:
    """Point a group member at their place beside the leader: straight at it when it is near, else
    at the next corner of a route to it (planned again every second)."""
    place = behaviours.place_of(human)
    human._destination, human._destination_blocked = place, False
    state = walkers.setdefault(human.id, {"planned": -99, "waypoints": [place], "index": 0})
    here = np.array([human.x, human.y])
    if math.dist(here, place) <= behaviours.FAR_FROM_PLACE_M:
        target = place
    else:
        if env.ticks - state["planned"] >= int(round(1.0 / env.TIMESTEP)):
            state["waypoints"], state["index"], state["planned"] = _plan_route(env, human, place), 0, env.ticks
        waypoints = state["waypoints"]
        while state["index"] < len(waypoints) - 1 and np.linalg.norm(here - waypoints[state["index"]]) < WAYPOINT_REACHED_M:
            state["index"] += 1
        target = waypoints[state["index"]]
    human.goal_x, human.goal_y = float(target[0]), float(target[1])


def has_reached_destination(human, offset=None) -> bool:
    """Replaces SocNavGym's Human.has_reached_goal: the final destination, not the next corner of
    the route, no giving up after 50 s of real time, and a wait there first for people who stop on
    arriving. Group members never choose a destination of their own."""
    if human.type == "static":
        return False
    traits = behaviours.traits_of(human)
    if traits is not None and traits.leader is not None:
        return False
    if offset is None:
        offset = human.width / 2
    gx, gy = getattr(human, "_destination", (human.goal_x, human.goal_y))
    distance = math.dist((human.x, human.y), (gx, gy))
    reached = distance < offset + human.goal_radius or \
        (getattr(human, "_destination_blocked", False) and distance < ARRIVED_NEXT_TO_CHAIR_M)
    return behaviours.arrived(human, reached)


# --------------------------------------------------------------------------- walking
def velocity(env, human, social: np.ndarray) -> np.ndarray:
    """This step's velocity (m/s) for a walking person. `social` is the social force model's
    velocity without the chair: towards the next corner of their route, away from walls,
    furniture and other people."""
    top = getattr(human, "_walk_speed", env.MAX_ADVANCE_HUMAN)
    traits = behaviours.traits_of(human)
    if traits is not None and traits.leader is not None:   # walking with a group: keep their place
        top = min(top * 1.25, env.MAX_ADVANCE_HUMAN)
        v = behaviours.follow(env, human, (human.goal_x, human.goal_y))
    else:
        destination = getattr(human, "_destination", None)
        if destination is not None:   # slow down when arriving
            top = min(top, max(0.3, math.dist((human.x, human.y), destination)))
        top *= behaviours.wait_for_group(human)
        v = _limit(social, top)
    v = behaviours.hold(env, human, v)
    v = _pass_chair(env, human, v, top)
    v = behaviours.sway(env, human, v)
    v = _accelerate(human, v, ACCELERATION * env.TIMESTEP)
    v = _not_into_chair(env, human, v)
    v = _along_walls(env, human, v)
    v = _limit(v, env.MAX_ADVANCE_HUMAN)
    human._last_velocity = v
    if np.linalg.norm(v) < 1e-3:   # standing: keep facing the same way
        return 1e-4 * np.array([math.cos(human.orientation), math.sin(human.orientation)])
    return v


def _limit(v, top) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    return v * (top / speed) if speed > top else v


def _accelerate(human, v, max_change) -> np.ndarray:
    """Change velocity gradually, so people speed up, slow down and turn smoothly."""
    previous = getattr(human, "_last_velocity", None)
    if previous is None:
        return v
    change = v - previous
    size = float(np.linalg.norm(change))
    return previous + change * (max_change / size) if size > max_change else v


def _along_walls(env, human, v) -> np.ndarray:
    """Never walk into a wall or furniture: next to one, the part of the velocity heading into it
    is dropped, so people slide along it (to get round the chair, for example) instead of
    pushing against it."""
    plan = plan_of(env)
    if plan.clearance_at(human.x, human.y) > human.width / 2 + WALL_SLIDE_M:
        return v
    n = np.asarray(plan.away_from_wall(human.x, human.y))
    into = float(v @ n)
    return v - into * n if into < 0 else v


def _has_room(env, point, human) -> bool:
    return plan_of(env).clearance_at(float(point[0]), float(point[1])) > human.width / 2 * 0.8


def _pass_chair(env, human, v, top) -> np.ndarray:
    """Walk past the chair with room to spare, on a side chosen once, or step aside and wait if
    there is no room to pass."""
    robot = env.robot
    to_chair = np.array([robot.x - human.x, robot.y - human.y])
    dist = float(np.linalg.norm(to_chair))
    attentive = human.avoids_robot
    look = ATTENTIVE_LOOK_S if attentive else DISTRACTED_LOOK_S
    reach = robot.radius + human.width / 2 + (ATTENTIVE_GAP_M if attentive else DISTRACTED_GAP_M)
    chair_v = robot.vel_x * np.array([math.cos(robot.orientation), math.sin(robot.orientation)])
    v_rel = v - chair_v
    speed_rel = float(np.linalg.norm(v_rel))
    if dist < 1e-6 or dist > reach + look * max(speed_rel, 0.5) + 0.5:
        human._pass_side = None
        return v
    u = to_chair / dist
    facing = v / np.linalg.norm(v) if np.linalg.norm(v) > 0.1 else \
        np.array([math.cos(human.orientation), math.sin(human.orientation)])
    if u @ facing < -0.3 and dist > reach + 0.3:     # the chair is behind them: they do not see it
        return v
    closing = to_chair @ v_rel > 0
    t = min(float(to_chair @ v_rel) / max(speed_rel ** 2, 1e-9), look) if closing else 0.0
    miss = float(np.linalg.norm(to_chair - v_rel * t))
    if human._pass_side is None:
        if not closing or miss >= reach:
            return v                                 # their path already clears the chair
        human._pass_side = _choose_side(env, human, u, v_rel, reach, dist)
    elif not closing and dist > reach:
        human._pass_side = None                      # past the chair
        return v
    side = human._pass_side
    if side != 0:   # if going round on that side would walk into a wall, try the other side
        here = np.array([human.x, human.y])
        for candidate in (side, -side, 0):
            if candidate == 0 or _has_room(env, here + _around(u, dist, reach, candidate) * 0.4, human):
                break
        side = human._pass_side = candidate
    if side == 0:
        if np.linalg.norm(chair_v) < 0.1:
            # The chair is waiting too: rather than both waiting, take another way round it.
            human._pass_side = None
            human._replan = True
            return v
        return _step_aside(env, human, u, chair_v)
    if dist <= reach:
        # Already closer than they like: keep walking towards their destination, but turn the
        # part of it heading into the chair into a step round it, and ease away a little.
        inward = float(v_rel @ u)
        if inward > 0:
            v = v - inward * u + inward * _rot(u, side * math.pi / 2)
        v = v - u * EASE_AWAY_SPEED * (reach - dist) / reach
        return _limit(v, top)
    v_new = chair_v + _around(u, dist, reach, side) * max(speed_rel, 0.3 * top)
    return _limit(v_new, top)


def _around(u, dist, reach, side) -> np.ndarray:
    """Direction towards the edge of the circle of `reach` around the chair, on the given side
    (or round the chair, when already inside the circle)."""
    angle = math.asin(min(reach / dist, 1.0))   # pi/2 (round the chair) once inside it
    return _rot(u, side * angle)


def _choose_side(env, human, u, v_rel, reach, dist) -> int:
    """+1 to pass with the chair on their right, -1 on their left, 0 if neither side has room.
    The natural side is the one their path already leans to."""
    cross = u[0] * v_rel[1] - u[1] * v_rel[0]
    side = 1 if cross > 1e-3 else -1 if cross < -1e-3 else (1 if human.id % 2 else -1)
    chair = np.array([env.robot.x, env.robot.y])
    for candidate in (side, -side):
        if _has_room(env, chair + _rot(u, candidate * math.pi / 2) * reach, human):
            return candidate
    return 0


def _step_aside(env, human, u, chair_v) -> np.ndarray:
    """No room to pass: move sideways out of the chair's way, as far as the walls allow, and wait.
    Back away slowly if the chair keeps coming."""
    here = np.array([human.x, human.y])
    best, best_room = np.zeros(2), plan_of(env).clearance_at(human.x, human.y)
    for side in (1, -1):
        direction = _rot(u, side * math.pi / 2)
        point = here + direction * 0.3
        room = plan_of(env).clearance_at(float(point[0]), float(point[1]))
        if room > best_room + 0.02 and room > human.width / 2:
            best, best_room = direction * YIELD_SPEED, room
    if chair_v @ (-u) > 0.1 and _has_room(env, here - u * 0.3, human):   # chair coming towards them
        best = best - u * YIELD_SPEED * 0.75
    return best


def _not_into_chair(env, human, v) -> np.ndarray:
    """Last resort: never take a step into the chair. The part of the velocity heading into it
    becomes a step around it."""
    robot = env.robot
    away = np.array([human.x - robot.x, human.y - robot.y])
    dist = float(np.linalg.norm(away))
    if dist < 1e-6:
        return v
    e = away / dist
    gap = dist - robot.radius - human.width / 2
    towards = -float(v @ e)
    if towards <= 0 or gap >= 0.3:
        return v
    side = human._pass_side or (1 if human.id % 2 else -1)
    removed = (1 - max(gap, 0.0) / 0.3) * towards
    return v + removed * e + removed * side * np.array([e[1], -e[0]])
