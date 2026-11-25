"""The people walking around the chair in Webots.

They walk exactly as in the 2D simulator: envs/pedestrians.py decides their routes,
how they pass the chair (attentive or distracted), when they step aside and wait,
and how smoothly they speed up and turn, and envs/behaviours.py gives each person a
character (on the phone, stopping now and then, walking in a group). The only part that is new here is the
"social force" SocNavGym used to provide underneath: a pull towards the next corner
of their route and a push away from other people. Webots then only draws them, with
their arms and legs swinging.

People are moved directly to their new position each step (the simulator does not
push them around with physics), as in the 2D simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from envs import behaviours, pedestrians
from envs.floorplan import ROUTE_CELL_M

PERSON_WIDTH_M = 0.72        # shoulder width, as human_diameter in envs/configs/base.yaml
ASIDE_SPEED = 0.8            # m/s: a walker the chair asks to make way steps aside this fast ...
ASIDE_STEPS_M = (0.6, 1.0, 1.5, 2.0, 2.5, 3.0)   # ... to the nearest spot clear of its way, this far at most ...
ASIDE_WAIT_S = 8.0           # ... and waits there until the chair has passed (they are behind it), this long at most
ASIDE_STUCK_S = 1.5          # ... or walks on if they get no more than 20 cm closer to it in this long (someone in the way)
CHAIR_RADIUS_M = 0.4         # as bridge.py's
CHAIR_CLEARANCE_M = 0.45     # the chair's centre stays this far from walls when it plans (envs/floorplan.py)
LEAVE_GAP_M = 0.3            # people standing still leave the chair this much room to pass them (edge to edge):
                             # its usual 15 cm, and as much again for steering round them in a tight spot ...
LEAVE_ROOM_M = 3.0           # ... never closing a way the chair had within this of them
WALK_SPEED = 1.0             # m/s, a walker's speed unless set otherwise (behaviours.py draws it)
ATTENTIVE_SHARE = 0.7        # prob_to_avoid_robot
PUSH_RANGE_M = 0.3           # people keep about this much room between each other ...
PUSH_SPEED = 0.6             # ... pushed apart at up to this speed (m/s) when closer
STOP_SHORT_M = 0.02          # people (and forklifts) walking up to the chair stop this far short of it
REPLAY_WAIT_S = 4.0          # a replayed person held up by someone this long gives up the recording and walks on
REPLAY_MAX_MPS = 2.0         # a replayed person never moves faster than this (no jumping to catch up with the recording)
REPLAY_GAP_M = 0.15          # a replayed person stops rather than come closer than this to anyone's edge
BUMP_M = 0.01                # someone walked from a phone can walk into the chair: they end this far into it, a hit
PRESSED_GAP_M = 0.15         # a walker this close to the chair (edge to edge) ...
PRESSED_SPEED = 0.3          # ... and getting along slower than this (m/s) takes another way round it
                             # (people only: a forklift or trolley never makes way)

# Events (bridge.py: the "events" test crowd, and training variety)
ELDERLY_SPEED = (0.4, 0.7)   # m/s: older people walk at about half speed ...
ELDERLY_ATTENTIVE = 0.4      # ... fewer of them notice the chair in time ...
ELDERLY_PAUSE_PER_S = 0.05   # ... and now and then stop for a moment (chance per second) ...
ELDERLY_PAUSE_S = (1.5, 4.0) # ... for this long
EXIT_SEARCH_M = 5.0          # someone stepping out of a room waits out of sight this close to the route
EXIT_MAX_WALK_M = 9.0        # ... no further than this walk from where they cross it

# Streaming (stream(), used by run.py): people appear and disappear around the chair, so there are always
# people where it is going, also after its goal changes.
STREAM_EVERY_S = 1.0         # how often the crowd is topped up and thinned out
ACTIVE_M = 14.0              # people this close to the chair count as being around it ...
ROUTE_AHEAD_M = 25.0         # ... and so do people near this much of the route ahead of it
ROUTE_CORRIDOR_M = 6.0       # (this close to it)
ROUTE_SHARE = 0.6            # this share of the crowd is kept on the route ahead, the rest around the chair
SPAWN_MIN_M = 8.0            # new people appear at least this far from the chair ...
SPAWN_RELAXED_M = 12.0       # ... and out of its sight, unless they are further than this
SEEN_M = 16.0                # (people this far from the chair count as out of sight anyway)
DESPAWN_M = 24.0             # people further than this, and out of its sight, are taken away again
DESPAWN_SEEN_M = 35.0        # ... and any that are in sight, once they are this far
MAX_DENSITY = 0.05           # people per square metre of the area around the chair, at most (corridors stay quiet)
ONWARD_MIN_M = 6.0           # someone who has arrived walks on to somewhere at least this far, ahead of them
ONWARD_COS = 0.5             # ... within 60 degrees of the way they were going (no turning back in a corridor)
TRANSIT_MIN_M = 12.0         # a person who just walks through the area heads for a spot at least this far away
SPAWN_PER_CALL = 1           # new groups per call (all that are missing when the goal changes)
GROUP_SIZE_MEAN = 2.3        # pairs and threes
# People walked by hand from phones (players.py)
PLAYER_WALK_SPEED = 1.3      # m/s forward, a brisk walk
PLAYER_BACK_SPEED = 0.5      # m/s stepping back
PLAYER_SIDE_SPEED = 0.8      # m/s stepping sideways
PLAYER_PICK_M = 15.0         # a player gets a random walker this close to the chair (anyone, if nobody is)

# Things that are not ordinary adults. Each moves (or stands) like a person in the crowd code, with
# its own size: collisions and the people tracker treat it as a circle `width` across. It is drawn as
# a box of (length, breadth, height) in colour, in one of the world's OBJ slots (make_world.py).
OBJECTS = {
    #           width  length breadth height   colour            speed m/s     notices chair
    "pet":      (0.45, (0.6, 0.25, 0.45), "0.45 0.3 0.15", (1.0, 1.5), True),
    "child":    (0.45, (0.3, 0.3, 1.1), "0.9 0.3 0.3", (1.4, 2.0), False),
    "trolley":  (1.4, (2.0, 0.8, 1.0), "0.85 0.85 0.9", (1.5, 2.0), False),
    "forklift": (1.8, (2.5, 1.2, 2.0), "0.95 0.75 0.1", (2.0, 2.5), False),
    "pallet":   (1.4, (1.2, 1.0, 1.1), "0.6 0.45 0.25", (0.0, 0.0), False),
}
SLOW_PATIENT_SPEED = (0.3, 0.5)   # m/s: patients with a drip stand or walking frame ...
SLOW_PATIENT_WIDTH = 0.9          # ... which makes them wider than a person


@dataclass
class Person:
    """One person, with the attributes envs/pedestrians.py reads (as SocNavGym's Human)."""
    id: int
    x: float
    y: float
    orientation: float = 0.0
    type: str = "dynamic"                  # "dynamic" walks, "static" stands still
    avoids_robot: bool = True              # attentive (True) or distracted
    width: float = PERSON_WIDTH_M
    goal_x: float | None = None            # the next corner of their route (set by pedestrians.py)
    goal_y: float | None = None
    goal_radius: float = 0.25
    speed: float = 0.0
    _walk_speed: float = WALK_SPEED
    _pass_side: int | None = None
    _last_velocity: np.ndarray | None = None
    _replan: bool = False
    gait: float = field(default=0.0, repr=False)   # how far through the walking animation (in steps)
    kind: str = "person"                   # or one of OBJECTS (a pet, a child, a trolley, ...)
    elderly: bool = False                  # walks slowly, notices the chair late, stops now and then
    _wander_left: float = 0.0              # pets and children: seconds until they change their mind
    _traits: object = None                 # envs/behaviours.py: phone, stops, waits, group
    slot: int | None = None                # which of the world's pedestrian models draws them (bridge.py)
    player: int | None = None              # 1 to 4: walked from a phone (take_over), not by the crowd
    by_phone: bool = False                 # ... by a person on a phone (crossing.py's scripted walker is not)
    _player_move: tuple = (0.0, 0.0)       # (forward m/s, turn rad/s), from players.py or crossing.py
    _player_side: float = 0.0              # m/s to their left
    _transit: bool = False                 # streamed in: walks through the area and leaves out of sight
    _stream: tuple | None = None           # (end A, end B): walks up and down between them
    _exit: tuple | None = None             # stepping out of a room: (crossing point, trigger distance,
                                           # destination); waits out of sight until the chair is near
    _aside: tuple | None = None            # a walker the chair asked to make way: stepping aside to here ...
    _aside_left: float = 0.0               # ... and waiting there, at most this many seconds more
    _aside_best: float = 0.0               # ... the closest they have got to the spot so far
    _aside_stuck: float = 0.0              # ... and for how many seconds they have got no nearer
    _replay: object = None                 # replay.py: following a recorded (SiT) track instead of the crowd's rules


def _line_gap(p, points) -> float:
    """Distance from point p to the polyline `points`."""
    if len(points) < 2:
        return math.dist(p, points[0]) if points else math.inf
    return min(_segment_gap(p, a, b) for a, b in zip(points, points[1:]))


def _closes_a_way(plan, spots, others, step_m: float = 0.1) -> bool:
    """Whether people standing at `spots` (with `others`, (x, y, radius), standing too) would close a
    way the chair could take before, within LEAVE_ROOM_M of them: a way with CHAIR_CLEARANCE_M for
    its centre from walls and LEAVE_GAP_M from its edge to theirs. Looks at where the chair's centre
    can be around them, before and after, and whether places on the edge of that area that were
    connected are not any more."""
    if not spots:
        return False
    cx, cy = sum(q[0] for q in spots) / len(spots), sum(q[1] for q in spots) / len(spots)
    n = int(2 * LEAVE_ROOM_M / step_m) + 1
    xs, ys = cx - LEAVE_ROOM_M + step_m * np.arange(n), cy + LEAVE_ROOM_M - step_m * np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    rows = np.clip(((plan.height_m / 2 - Y) / plan.resolution).astype(int), 0, plan.clearance.shape[0] - 1)
    cols = np.clip(((X + plan.width_m / 2) / plan.resolution).astype(int), 0, plan.clearance.shape[1] - 1)
    before = (np.abs(X) < plan.width_m / 2) & (np.abs(Y) < plan.height_m / 2) & \
        (plan.clearance[rows, cols] >= CHAIR_CLEARANCE_M)
    for ox, oy, r in others:
        before &= (X - ox) ** 2 + (Y - oy) ** 2 >= (r + LEAVE_GAP_M + CHAIR_RADIUS_M) ** 2
    after = before.copy()
    for q in spots:                      # (x, y), or (x, y, radius) for something other than a person
        r = q[2] if len(q) > 2 else PERSON_WIDTH_M / 2
        after &= (X - q[0]) ** 2 + (Y - q[1]) ** 2 >= (r + LEAVE_GAP_M + CHAIR_RADIUS_M) ** 2
    lab_before, lab_after = _label(before), _label(after)
    edge = np.zeros_like(before)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    for k in np.unique(lab_before[edge & before]):
        joined = np.unique(lab_after[edge & (lab_before == k) & after])
        if len(joined[joined > 0]) > 1:
            return True
    return False


def _label(free: np.ndarray) -> np.ndarray:
    """Connected areas of True cells (4 neighbours), numbered from 1; 0 where False."""
    lab = np.zeros(free.shape, dtype=np.int32)
    rows, cols = free.shape
    count = 0
    for r0, c0 in zip(*np.nonzero(free)):
        if lab[r0, c0]:
            continue
        count += 1
        lab[r0, c0] = count
        todo = [(r0, c0)]
        while todo:
            r, c = todo.pop()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < rows and 0 <= cc < cols and free[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = count
                    todo.append((rr, cc))
    return lab


def _segment_gap(p, a, b) -> float:
    """Distance from point p to the segment a-b."""
    length2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    t = 0.0 if length2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length2))
    return math.dist(p, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))


@dataclass
class Destination:
    x: float
    y: float


class Crowd:
    """Everyone in the scene. `scene` is the WebotsScene: pedestrians.py reads the chair, the
    floor plan and the time step from it, as it does from SocNavGym's environment."""

    def __init__(self, scene):
        self.scene = scene
        self.walking: list[Person] = []
        self.standing: list[Person] = []
        self.rng = random.Random(0)
        self.cfg = behaviours.settings()
        self.scale = 1.0                       # stream(): people around the chair, relative to a trip's usual crowd
        self.base_walkers = 0                  # walking people a trip starts with (before any scaling)
        self._zone = None                      # stream(): spawn cells around the chair and its route
        self._goal = (0.0, 0.0)
        self.learned = None                    # learned_walk.LearnedWalkers with --walkers learned

    @property
    def everyone(self) -> list[Person]:
        return self.walking + self.standing

    # ------------------------------------------------------------ placing people
    def reset(self, rng: random.Random, n_walking: int, n_standing: int, n_groups: int,
              area: np.ndarray, avoid: list[tuple[float, float]], attentive_share: float = ATTENTIVE_SHARE,
              elderly_share: float = 0.0) -> None:
        """Place people at random spots in `area` (route-grid cells), away from the points in
        `avoid` (the chair's start and goal) and from each other."""
        self.rng = rng
        scene, plan = self.scene, self.scene.plan
        taken: list[tuple[float, float]] = []

        def free_spot(room: float, ok=lambda spot: True):
            for _ in range(200):
                x, y = plan.random_point(rng, area)
                if all(math.dist((x, y), p) > 1.5 for p in avoid) and \
                        all(math.dist((x, y), p) > room for p in taken) and ok((x, y)):
                    taken.append((x, y))
                    return x, y
            return None

        def leaves_room(spots) -> bool:   # people stopping to talk or wait leave the chair a way past
            return not _closes_a_way(plan, spots, [(o.x, o.y, o.width / 2) for o in self.standing])

        next_id = 0
        self.walking, self.standing = [], []
        for _ in range(n_walking):
            spot = free_spot(1.0)
            if spot is None:
                break
            self._new_walker(rng, next_id, spot, area, attentive_share, elderly_share)
            next_id += 1
        for _ in range(n_standing):
            spot = free_spot(1.0, lambda q: leaves_room([q]))
            if spot is None:
                break
            self.standing.append(Person(next_id, *spot, orientation=rng.uniform(-math.pi, math.pi),
                                        type="static"))
            next_id += 1
        for _ in range(n_groups):   # 2 or 3 people standing in a circle, talking
            group = {}

            def members(centre):
                if centre not in group:
                    size = rng.randint(2, 3)
                    turn = rng.uniform(0, 2 * math.pi)
                    spots = [(centre[0] + 0.55 * math.cos(turn + 2 * math.pi * k / size),
                              centre[1] + 0.55 * math.sin(turn + 2 * math.pi * k / size), turn + 2 * math.pi * k / size)
                             for k in range(size)]
                    group[centre] = [q for q in spots if plan.clearance_at(q[0], q[1]) > PERSON_WIDTH_M / 2]
                return group[centre]

            centre = free_spot(2.0, lambda c: leaves_room([q[:2] for q in members(c)]))
            if centre is None:
                break
            for x, y, a in members(centre):
                self.standing.append(Person(next_id, x, y, orientation=a + math.pi, type="static"))
                next_id += 1
        # Everyone else gets a character: some on the phone, some stopping, some walking in groups.
        behaviours.assign(scene, [p for p in self.walking if not p.elderly], rng, self.cfg, plan)
        scene._walkers = {}
        pedestrians.before_step(scene)

    def _new_walker(self, rng: random.Random, person_id: int, spot, area, attentive_share: float,
                    elderly_share: float) -> Person:
        """A walking person at `spot`, heading for a random destination in `area` (not yet given a
        character beyond an older person's: see behaviours.assign)."""
        destination = Destination(*self.scene.plan.random_point(rng, area))
        person = Person(person_id, *spot, orientation=rng.uniform(-math.pi, math.pi),
                        avoids_robot=rng.random() < attentive_share,
                        goal_x=destination.x, goal_y=destination.y,
                        _walk_speed=behaviours.draw_speed(rng, self.cfg))
        if elderly_share and rng.random() < elderly_share:
            person.elderly = True
            person._walk_speed = rng.uniform(*ELDERLY_SPEED)
            person.avoids_robot = rng.random() < ELDERLY_ATTENTIVE
            behaviours.give_traits(person, rng, self.cfg)
            person._traits.pause_rate, person._traits.pause_seconds = ELDERLY_PAUSE_PER_S, ELDERLY_PAUSE_S
        self.walking.append(person)
        self.scene.goals[person.id] = destination
        return person

    def add_room_exits(self, rng: random.Random, route: list[tuple[float, float]], n: int, group: int = 1,
                       limit: int = 10**6) -> int:
        """Add up to n people who wait out of sight (in a room, behind a wall) near the chair's route,
        and step out across it when the chair comes close. With group > 1, each is a group leaving a
        room together (a meeting ending). Never more than `limit` people. Returns how many were placed."""
        from envs.floorplan import route_length
        plan = self.scene.plan
        points = _along(route, 0.5)
        total = route_length(route[0], route[1:])
        crossings = [p for k, p in enumerate(points) if 4.0 <= k * 0.5 <= total - 2.0]
        placed = 0
        next_id = max((p.id for p in self.everyone), default=-1) + 1

        def clear_of_route(spot):   # not standing on the route (behind a plant or shelf), nor near the start
            return min(math.dist(spot, q) for q in points) > 1.3 and math.dist(spot, route[0]) > 3.0

        for _ in range(n if crossings else 0):
            for _attempt in range(40):
                cross = rng.choice(crossings)
                hide = plan.random_point(rng, plan.cells_near([cross], EXIT_SEARCH_M))
                if math.dist(hide, cross) < 1.5 or line_of_sight(plan, hide, cross) or not clear_of_route(hide):
                    continue                    # must be out of sight of the route, and off it
                if any(math.dist(hide, (o.x, o.y)) < 1.0 for o in self.everyone):
                    continue
                way = plan.route(hide, cross, for_people=True)
                if not way or route_length(hide, way) > EXIT_MAX_WALK_M:
                    continue
                walk = route_length(hide, way)
                for _try in range(20):          # where they are going: across the route, in view of it
                    dest = plan.random_point(rng, plan.cells_near([cross], 4.0))
                    if line_of_sight(plan, cross, dest) and math.dist(dest, hide) > walk * 0.5 + 1.0:
                        break
                else:
                    continue
                spots = [hide]
                for _try in range(30 * group):  # the rest of the group: next to them, also out of sight
                    if len(spots) >= group:
                        break
                    a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.8, 1.8)
                    spot = (hide[0] + r * math.cos(a), hide[1] + r * math.sin(a))
                    if plan.clearance_at(*spot) > PERSON_WIDTH_M / 2 + 0.05 and \
                            not line_of_sight(plan, spot, cross) and line_of_sight(plan, spot, hide) and \
                            all(math.dist(spot, q) > 0.8 for q in spots) and clear_of_route(spot):
                        spots.append(spot)
                for spot in spots[:max(0, limit - placed)]:
                    others = (rng.uniform(-2, 2), rng.uniform(-2, 2)) if len(spots) > 1 else (0.0, 0.0)
                    goal = dest if not plan.clearance_at(dest[0] + others[0], dest[1] + others[1]) > 0.5 \
                        else (dest[0] + others[0], dest[1] + others[1])
                    person = Person(next_id, *spot, orientation=math.atan2(cross[1] - spot[1], cross[0] - spot[0]),
                                    type="static", avoids_robot=rng.random() < 0.5,
                                    _walk_speed=behaviours.draw_speed(rng, self.cfg))
                    person._exit = (cross, walk * 1.3 + 1.5, goal)
                    self.standing.append(person)
                    next_id += 1
                    placed += 1
                break
        return placed

    def add_waiting_area(self, rng: random.Random, route: list[tuple[float, float]], size: int,
                         limit: int = 10**6) -> int:
        """A cluster of `size` people standing (waiting, talking) 2 to 5 m from the chair's route,
        never on it. Returns how many were placed."""
        plan = self.scene.plan
        whole = _along(route, 1.0)          # (the whole route, start and goal included, stays clear)
        points = whole[2:-2]
        next_id = max((p.id for p in self.everyone), default=-1) + 1
        for _attempt in range(40 if points else 0):
            near = rng.choice(points)
            centre = plan.random_point(rng, plan.cells_near([near], 5.0))
            if min(math.dist(centre, q) for q in whole) < 2.0 + 0.5:
                continue
            spots = [centre]
            for _try in range(40):
                if len(spots) >= min(size, limit):
                    break
                a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.8)
                spot = (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a))
                if plan.clearance_at(*spot) > PERSON_WIDTH_M / 2 and all(math.dist(spot, q) > 0.75 for q in spots) \
                        and min(math.dist(spot, q) for q in whole) > 1.6:
                    spots.append(spot)
            for spot in spots:
                self.standing.append(Person(next_id, *spot, orientation=rng.uniform(-math.pi, math.pi), type="static"))
                next_id += 1
            return len(spots)
        return 0

    def make_phone_walkers(self, rng: random.Random, n: int) -> None:
        """n more walkers (on top of the usual share) become people on the phone: slow, not
        watching, drifting from side to side."""
        for person in [p for p in self.walking if not p.elderly and not behaviours.is_phone(p)
                       and not behaviours.in_group(p)][:n]:
            behaviours.make_phone(person, rng, self.cfg)

    def make_carriers(self, rng: random.Random, n: int) -> None:
        """n walkers carry boxes in front of them: they do not see the chair in time."""
        for person in [p for p in self.walking if not p.elderly and not behaviours.is_phone(p)][-n:] if n else []:
            person.avoids_robot = False

    def make_streams(self, rng: random.Random, route: list[tuple[float, float]], share: float) -> None:
        """A share of the walkers walk up and down along the chair's route, both ways (two-way traffic)."""
        if len(route) < 2:
            return
        points = _along(route, 0.5)
        ends = (points[len(points) * 15 // 100], points[len(points) * 85 // 100])   # short of the chair's start and goal
        for k, person in enumerate(self.walking):
            if rng.random() < share and not behaviours.in_group(person):
                a, b = ends if k % 2 else ends[::-1]
                person._stream = (b, a)
                if behaviours.traits_of(person):
                    person._traits.dwell_chance = 0.0       # keeps walking up and down
                person.goal_x, person.goal_y = b
                self.scene.goals[person.id] = Destination(*b)

    def make_slow_patients(self, rng: random.Random, n: int) -> None:
        """n walkers become patients with a drip stand or walking frame: very slow and wider."""
        for person in [p for p in self.walking if p.kind == "person" and not behaviours.is_phone(p)
                  and not behaviours.in_group(p)][:n]:
            person._walk_speed = rng.uniform(*SLOW_PATIENT_SPEED)
            person.width = SLOW_PATIENT_WIDTH

    def add_object(self, rng: random.Random, kind: str, route: list[tuple[float, float]]) -> Person | None:
        """Add a pet, child, trolley, forklift or pallet (OBJECTS) where it will meet the chair:
          pet, child   wander near the middle of the chair's route, changing direction often
          trolley, forklift   come towards the chair along its route from far ahead, and back
          pallet       stands beside the route (not on the map: only the sensors find it)"""
        plan = self.scene.plan
        width, _box, _colour, speed, notices = OBJECTS[kind]
        points = _along(route, 0.5)
        if len(points) < 8:
            return None
        next_id = max((p.id for p in self.everyone), default=-1) + 1
        middle = points[len(points) // 3: 2 * len(points) // 3]
        for _attempt in range(40):
            if kind in ("pet", "child"):
                spot = plan.random_point(rng, plan.cells_near([rng.choice(middle)], 3.0))
                ok = plan.clearance_at(*spot) > width / 2 + 0.05
            elif kind == "pallet":       # beside the route, leaving the chair at least 1.2 m to pass
                a, b = middle[rng.randrange(len(middle) - 1)], None
                k = points.index(a)
                b = points[min(k + 1, len(points) - 1)]
                heading = math.atan2(b[1] - a[1], b[0] - a[0])
                side = rng.choice([-1, 1]) * rng.uniform(0.9, 1.6)
                spot = (a[0] - math.sin(heading) * side, a[1] + math.cos(heading) * side)
                room = plan.clearance_at(*spot)
                ok = room > width / 2 and plan.clearance_at(*a) > 0.6 and \
                    min(math.dist(spot, q) for q in points) > width / 2 + 0.4 + 0.4 and \
                    not _closes_a_way(plan, [(*spot, width / 2)], [(o.x, o.y, o.width / 2) for o in self.standing])
            else:                        # trolley, forklift: start far along the route, out of sight
                spot = points[min(len(points) - 1, int(len(points) * rng.uniform(0.75, 0.95)))]
                ok = plan.clearance_at(*spot) > width / 2
            if ok and all(math.dist(spot, (o.x, o.y)) > (o.width + width) / 2 + 0.3 for o in self.everyone):
                break
        else:
            return None
        if kind == "pallet":
            thing = Person(next_id, *spot, orientation=rng.uniform(-math.pi, math.pi), type="static",
                           kind=kind, width=width)
            self.standing.append(thing)
            return thing
        thing = Person(next_id, *spot, kind=kind, width=width, avoids_robot=notices,
                       _walk_speed=rng.uniform(*speed))
        if kind in ("trolley", "forklift"):
            back = points[int(len(points) * 0.05)]
            thing._stream = (back, spot)            # rushing towards the chair's start, then back
            goal = back
        else:
            goal = plan.random_point(rng, plan.cells_near([spot], 4.0))
        thing.goal_x, thing.goal_y = goal
        thing.orientation = math.atan2(goal[1] - spot[1], goal[0] - spot[0])
        self.walking.append(thing)
        self.scene.goals[thing.id] = Destination(*goal)
        return thing

    # ------------------------------------------------------------ streaming
    # ------------------------------------------------------------ people walked by hand
    def player(self, number: int) -> Person | None:
        return next((p for p in self.walking if p.player == number), None)

    def take_over(self, number: int, rng: random.Random, not_this=None) -> Person | None:
        """Give player `number` a random walker to steer: someone walking on their own (not in a group,
        not walking up and down a route on purpose), near the chair if anyone is. The crowd stops
        steering them, and everyone else, the chair included, treats them as an ordinary person."""
        chair = self.scene.robot
        def free(p):
            traits = p._traits
            return (p.kind == "person" and p.player is None and not p._stream and p._exit is None
                    and (traits is None or (traits.leader is None and not traits.members)))
        candidates = [p for p in self.walking if free(p) and p is not not_this]
        near = [p for p in candidates if math.dist((p.x, p.y), (chair.x, chair.y)) <= PLAYER_PICK_M]
        if not candidates:
            return None
        person = rng.choice(near or candidates)
        person.player, person._player_move, person._player_side, person._transit = number, (0.0, 0.0), 0.0, False
        person.by_phone = True
        person.goal_x = person.goal_y = None           # pedestrians.py plans no route for them
        self.scene._walkers.pop(person.id, None)
        return person

    def let_go(self, number: int) -> Person | None:
        """Hand player `number`'s walker back to the crowd: they walk on to somewhere new."""
        person = self.player(number)
        if person is not None:
            person.player, person._player_move, person._player_side, person.by_phone = None, (0.0, 0.0), 0.0, False
            destination = self._onward(person, self.scene.people_area)
            person.goal_x, person.goal_y = destination.x, destination.y
            self.scene.goals[person.id] = destination
        return person

    def stream(self, force: bool = False) -> None:
        """Keep people around the chair and where it is going. Walkers far from the chair and out of its
        sight are taken away (their pedestrian model is free for someone else), and when fewer people
        than a trip's crowd (times `scale`) are around the chair or along the route ahead, new ones
        appear out of its sight. `force` (the goal has just changed) fills it all in at once."""
        scene, plan, chair = self.scene, self.scene.plan, self.scene.robot
        here = (chair.x, chair.y)
        ahead = _route_ahead(scene.route, here, ROUTE_AHEAD_M)
        corridor = plan.cells_near(ahead, ROUTE_CORRIDOR_M)
        zone = _union(plan.cells_near([here], ACTIVE_M), corridor)
        if scene.crowd_style == "busy" and len(zone):        # (spread crowds roam the whole building)
            scene.people_area = zone
        self._zone, self._goal = zone, (chair.goal_x, chair.goal_y)
        self._take_away_far_people(here)
        slots = max(0, scene._free_slots() - 2)               # (a couple of models stay free for events)
        target = min(round(self.scale * self.base_walkers), density_cap(zone))
        walkers = [p for p in self.walking if p.kind == "person"]
        around = sum(1 for p in walkers if _is_around(p, here, ahead))
        on_route = sum(1 for p in walkers if _near_route(p, ahead))
        goal = (chair.goal_x, chair.goal_y)
        batches = 10**6 if force else SPAWN_PER_CALL
        while around < target and slots > 0 and batches > 0:
            batches -= 1
            share = self.cfg["group_share"]
            p_group = share / (GROUP_SIZE_MEAN * (1 - share) + share)
            size = 1 if self.rng.random() >= p_group else (3 if self.rng.random() < self.cfg["group_of_three_share"] else 2)
            size = min(size, slots, target - around)
            pool = corridor if on_route < ROUTE_SHARE * target and len(corridor) else zone
            made = self._spawn(size, pool, here, goal, zone)
            if not made and pool is corridor:        # no room out of sight on the route (it is short, or in view)
                made = self._spawn(size, zone, here, goal, zone)
            if not made:
                break
            slots -= len(made)
            around += len(made)
            on_route += sum(1 for p in made if _near_route(p, ahead))

    def _take_away_far_people(self, here) -> None:
        """Remove walkers who are far from the chair and out of its sight, a whole group at a time."""
        plan = self.scene.plan
        gone = []
        for p in self.walking:
            if p.kind != "person" or p._stream or p.player or p._replay is not None \
                    or p._traits is not None and p._traits.leader is not None:
                continue
            group = [p] + (p._traits.members if p._traits is not None else [])
            if all(self._far_and_unseen(q, here, plan) for q in group):
                gone += group
        self._remove(gone)

    def _remove(self, people) -> None:
        for p in people:
            if p in self.walking:
                self.walking.remove(p)
            self.scene.goals.pop(p.id, None)
            self.scene._walkers.pop(p.id, None)

    @staticmethod
    def _far_and_unseen(person, here, plan, min_distance: float = DESPAWN_M) -> bool:
        """Further than `min_distance` from `here` and out of its sight (or further than DESPAWN_SEEN_M)."""
        d = math.dist((person.x, person.y), here)
        return d > DESPAWN_SEEN_M or (d > min_distance and not line_of_sight(plan, (person.x, person.y), here))

    def _hidden_spot(self, zone, here, goal):
        """A free spot in `zone`, at least SPAWN_MIN_M from the chair and out of its sight (or well
        away from it), not on the people already there. None if there is none."""
        plan = self.scene.plan
        if len(zone) == 0:
            return None
        for attempt in range(80):
            x, y = plan.random_point(self.rng, zone)
            d = math.dist((x, y), here)
            near_enough = SPAWN_MIN_M if attempt < 60 else SPAWN_RELAXED_M
            if d < near_enough or math.dist((x, y), goal) < 2.0 or plan.clearance_at(x, y) < PERSON_WIDTH_M / 2 + 0.1:
                continue
            if d < (SEEN_M if attempt < 60 else SPAWN_RELAXED_M) and line_of_sight(plan, (x, y), here):
                continue
            if any(math.dist((x, y), (o.x, o.y)) < (o.width + PERSON_WIDTH_M) / 2 + 0.5 for o in self.everyone):
                continue
            return x, y
        return None

    def _spawn(self, size: int, pool, here, goal, area) -> list[Person]:
        """Add `size` people (a group if more than one) at spots of `pool` out of the chair's sight, walking
        to places in `area` (around the chair and its route). Returns the new people."""
        scene = self.scene
        trip = scene.trip
        next_id = max((p.id for p in self.everyone), default=-1) + 1
        made = []
        for _ in range(size):
            spot = self._hidden_spot(pool, here, goal)
            if spot is None:
                break
            person = self._new_walker(self.rng, next_id, spot, area, trip.attentive_share, trip.elderly_share)
            next_id += 1
            made.append(person)
        ordinary = [p for p in made if not p.elderly]
        cfg = {**self.cfg, "group_share": 1.0 if len(ordinary) > 1 else 0.0}
        behaviours.assign(scene, ordinary, self.rng, cfg, scene.plan)
        for person in made:                 # they walk through: somewhere far, out of sight, and then they are gone
            person._transit = True
            self._head_for_exit(person, here, goal)
        return made

    def _head_for_exit(self, person: Person, here, goal) -> None:
        """Send a streamed-in person towards a spot out of the chair's sight, far from where they are."""
        spot = None
        for _ in range(12):
            spot = self._hidden_spot(self._zone, here, goal)
            if spot is not None and math.dist(spot, (person.x, person.y)) >= TRANSIT_MIN_M:
                break
        if spot is None:
            spot = self.scene.plan.random_point(self.rng, self._zone)
        person.goal_x, person.goal_y = spot
        self.scene.goals[person.id] = Destination(*spot)

    def _arrived_in_transit(self, person: Person) -> None:
        """A streamed-in person has got where they were going: gone if out of the chair's sight (with
        their group), else on to another spot out of sight."""
        chair = self.scene.robot
        here = (chair.x, chair.y)
        group = [person] + (person._traits.members if person._traits is not None else [])
        hidden = all(self._far_and_unseen(q, here, self.scene.plan, min_distance=0.0) for q in group)
        if hidden:
            self._remove(group)
        else:
            self._head_for_exit(person, here, self._goal)

    def _onward(self, person: Person, area) -> Destination:
        """Where someone who has arrived goes next: not back where they came from, but on, ahead of
        them (people in a corridor walk along it, they do not pace up and down)."""
        plan = self.scene.plan
        c, s = math.cos(person.orientation), math.sin(person.orientation)
        for _ in range(40):
            x, y = plan.random_point(self.rng, area)
            d = math.dist((x, y), (person.x, person.y))
            if d >= ONWARD_MIN_M and ((x - person.x) * c + (y - person.y) * s) / d >= ONWARD_COS:
                return Destination(x, y)
        return Destination(*plan.random_point(self.rng, area))

    # ------------------------------------------------------------ walking
    def askable(self, x: float, y: float) -> bool:
        """Whether someone the chair could ask to make way is at (x, y): a person walking (also one who
        has stopped for a moment). Never people standing to talk or wait (they are gone round), an
        object such as a pallet or a forklift, or a person walked from a phone."""
        return self._walker_at(x, y) is not None

    def _walker_at(self, x: float, y: float) -> Person | None:
        near = [p for p in self.walking if p.kind == "person" and p.player is None
                and math.dist((p.x, p.y), (x, y)) < 0.5]
        return min(near, key=lambda p: math.dist((p.x, p.y), (x, y)), default=None)

    def make_way(self, x: float, y: float, chair_xy, lane) -> bool | None:
        """The chair asks the walker at (x, y) to let it past ("Excuse me"): they step to the nearest
        spot clear of the chair's way (`lane`: its route ahead) where there is room, walking round
        anyone in between, wait there until the chair has passed, then walk on. False if there is no
        such spot, None if no walker is there (people standing to talk or wait are never asked)."""
        plan = self.scene.plan
        p = self._walker_at(x, y)
        if p is None:
            return None
        if p._aside is not None:
            return True
        others = [o for o in self.everyone if o is not p]
        taken = [(o.x, o.y) for o in others]
        clear_of_lane = CHAIR_RADIUS_M + p.width / 2 + 0.2

        def walkable(a, b):              # a straight step from a to b: no wall, not past the chair, and
            n = max(2, int(math.dist(a, b) / 0.1))      # no closer to anyone than they are now or than is comfortable
            return all(plan.clearance_at(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) > 0.25
                       for i in range(1, n + 1)) and _segment_gap(chair_xy, a, b) > CHAIR_RADIUS_M + p.width / 2 + 0.15 \
                and all(_segment_gap((o.x, o.y), a, b) >= min(math.dist((o.x, o.y), a), (p.width + o.width) / 2 + 0.1) - 1e-6
                        for o in others)

        for d in ASIDE_STEPS_M:
            for k in range(16):
                q = (p.x + d * math.cos(2 * math.pi * k / 16), p.y + d * math.sin(2 * math.pi * k / 16))
                if self._room_to_stand(plan, q, p, lane, clear_of_lane, taken) and walkable((p.x, p.y), q):
                    p._aside, p._aside_left = q, ASIDE_WAIT_S
                    p._aside_best, p._aside_stuck = math.dist((p.x, p.y), q), 0.0
                    return True
        return False

    @staticmethod
    def _room_to_stand(plan, q, person, lane, clear_of_lane, taken) -> bool:
        """Whether `person` can stand at q: room from the walls, clear of the chair's way and of
        everyone else."""
        return plan.clearance_at(*q) > person.width / 2 + 0.05 and _line_gap(q, lane) > clear_of_lane and \
            all(math.dist(q, o) > PERSON_WIDTH_M + 0.05 for o in taken)

    def step(self, dt: float) -> None:
        """Move everyone on by dt seconds."""
        scene, plan = self.scene, self.scene.plan
        area = scene.people_area
        chair = scene.robot
        for person in [p for p in self.standing if p._exit]:   # the chair is near: step out of the room
            cross, trigger, dest = person._exit
            if math.dist(cross, (chair.x, chair.y)) < trigger:
                person._exit, person.type = None, "dynamic"
                person.goal_x, person.goal_y = dest
                self.standing.remove(person)
                self.walking.append(person)
                scene.goals[person.id] = Destination(*dest)
        for person in self.walking:            # pets and children change their mind every few seconds
            if person.kind in ("pet", "child"):
                person._wander_left -= dt
                if person._wander_left <= 0:
                    person._wander_left = self.rng.uniform(1.0, 3.0)
                    goal = plan.random_point(self.rng, plan.cells_near([(person.x, person.y)], 4.0))
                    person.goal_x, person.goal_y = goal
                    scene.goals[person.id] = Destination(*goal)
        for person in list(self.walking):      # arrived: a new destination
            if person.player is None and person._replay is None and pedestrians.has_reached_destination(person):
                if person._stream:             # walking up and down between two ends
                    person._stream = (person._stream[1], person._stream[0])
                    person.goal_x, person.goal_y = person._stream[0]
                    scene.goals[person.id] = Destination(*person._stream[0])
                elif person._transit:          # passing through: leaves once out of the chair's sight
                    self._arrived_in_transit(person)
                else:
                    scene.goals[person.id] = self._onward(person, area)
        pedestrians.before_step(scene)

        if self.learned is not None:           # --walkers learned: Trajectron++ suggests each step
            self.learned.update(self, dt)
        velocities = []
        for person in self.walking:
            if person._replay is not None:     # a recorded track: where it has them one step on
                v = person._replay.velocity(person, scene.ticks * dt, dt)
                if v is not None and self._blocked(person, v, dt):
                    person._replay.start += dt           # the recording has no one else in it: wait (it waits too)
                    person._replay.held += dt
                    if person._replay.held < REPLAY_WAIT_S:
                        velocities.append(np.zeros(2))
                        continue
                    v = None                             # still blocked after a few seconds: walk on as an ordinary walker
                elif v is not None:
                    person._replay.held = 0.0
                    speed = float(np.hypot(v[0], v[1]))
                    if speed > REPLAY_MAX_MPS:           # behind the recording (it was held up): catch up at a walk,
                        person._replay.start += dt * (1.0 - REPLAY_MAX_MPS / speed)   # the recording waiting for them
                        v = v * (REPLAY_MAX_MPS / speed)
                    velocities.append(v)
                    continue
                person._replay = None          # the recording is over (or they were held up): an ordinary walker from here
                scene.goals[person.id] = self._onward(person, area)
                person.goal_x, person.goal_y = scene.goals[person.id].x, scene.goals[person.id].y
            if person._aside is not None:      # asked by the chair to make way
                velocities.append(self._stepping_aside(person, dt))
                continue
            if person.player is not None:      # walked by hand: forward, sideways and turn
                forward, turn = person._player_move
                person.orientation = math.atan2(math.sin(person.orientation + turn * dt),
                                                math.cos(person.orientation + turn * dt))
                c, s = math.cos(person.orientation), math.sin(person.orientation)
                velocities.append(np.array([c, s]) * forward + np.array([-s, c]) * person._player_side)
                continue
            social = self.learned.velocity(person) if self.learned is not None else None
            if social is None:                 # the rules (always for pets, trolleys and so on)
                social = self._social(person)
            v = pedestrians.velocity(scene, person, social)
            velocities.append(v)
        for person, v in zip(self.walking, velocities):
            x, y = person.x + v[0] * dt, person.y + v[1] * dt
            x, y = plan.push_out(x, y, person.width / 2)
            chair = scene.robot
            touching = chair.radius + person.width / 2
            now, then = math.dist((person.x, person.y), (chair.x, chair.y)), math.dist((x, y), (chair.x, chair.y))
            if person.by_phone and then < touching:   # walked into the chair from a phone: they bump into it,
                k = (touching - BUMP_M) / max(now, 1e-6)  # which counts as a hit, as it would in real life
                x, y = chair.x + (person.x - chair.x) * k, chair.y + (person.y - chair.y) * k
            elif then < touching or (person.player is None and then < touching + STOP_SHORT_M and then < now):
                x, y = person.x, person.y      # never step into the chair (as in the 2D simulator), nor right up to it ...
            if person.player is None and person.kind == "person" and math.dist((x, y), (person.x, person.y)) < PRESSED_SPEED * dt and \
                    math.dist((x, y), (chair.x, chair.y)) - chair.radius - person.width / 2 < PRESSED_GAP_M:
                person._replan = True          # ... and, pressed against it, take another way round it
            moved = math.dist((x, y), (person.x, person.y))
            person.x, person.y = x, y
            person.speed = moved / dt
            if moved > 1e-4 and person.player is None:   # (a player's own turning sets theirs)
                person.orientation = math.atan2(v[1], v[0])
            person.gait += moved

    def _blocked(self, person: Person, v: np.ndarray, dt: float) -> bool:
        """Would this step of a recorded track take `person` into someone, or closer to them than REPLAY_GAP_M?"""
        nx, ny = person.x + v[0] * dt, person.y + v[1] * dt
        chair = self.scene.robot
        others = [(o.x, o.y, (o.width + person.width) / 2 + REPLAY_GAP_M) for o in self.everyone if o is not person]
        others.append((chair.x, chair.y, chair.radius + person.width / 2 + REPLAY_GAP_M))   # the chair too
        for ox, oy, reach in others:
            if math.dist((nx, ny), (ox, oy)) < reach and math.dist((nx, ny), (ox, oy)) < math.dist((person.x, person.y), (ox, oy)):
                return True
        return False

    def _stepping_aside(self, person: Person, dt: float) -> np.ndarray:
        """A walker the chair asked to make way: to their spot at ASIDE_SPEED, kept apart from everyone
        else on the way (not through them), then waiting there until the chair has passed or
        ASIDE_WAIT_S is up. Then they walk on to where they were going. If someone keeps them from
        getting there, they walk on after ASIDE_STUCK_S (the chair asks again later)."""
        chair = self.scene.robot
        gx, gy = person._aside
        d = math.dist((person.x, person.y), (gx, gy))
        person._aside_left -= dt
        if d < person._aside_best - 0.2:
            person._aside_best, person._aside_stuck = d, 0.0
        elif d >= 0.1:
            person._aside_stuck += dt
        behind = (person.x - chair.x) * math.cos(chair.orientation) + \
            (person.y - chair.y) * math.sin(chair.orientation) < -(CHAIR_RADIUS_M + person.width / 2)
        # the chair has gone past them, they waited long enough, or they cannot get to the spot
        if person._aside_left <= 0 or behind or person._aside_stuck > ASIDE_STUCK_S:
            person._aside = None
            return np.zeros(2)
        v = np.zeros(2) if d < 0.1 else np.array([gx - person.x, gy - person.y]) / d * min(ASIDE_SPEED, d / dt)
        v = pedestrians._accelerate(person, v + self._push(person), pedestrians.ACCELERATION * dt)
        v = pedestrians._along_walls(self.scene, person, pedestrians._not_into_chair(self.scene, person, v))
        person._last_velocity = v                 # (walking as everyone does: round the chair, along walls)
        return v

    def _social(self, person: Person) -> np.ndarray:
        """Towards the next corner of their route at their own walking speed, pushed away from
        anyone closer than PUSH_RANGE_M (edge to edge)."""
        v = np.zeros(2)
        if person.goal_x is not None:
            d = np.array([person.goal_x - person.x, person.goal_y - person.y])
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                v = d / n * person._walk_speed
        return v + self._push(person)

    def _push(self, person: Person) -> np.ndarray:
        """Away from anyone closer than PUSH_RANGE_M (edge to edge), so people do not walk into each other."""
        v = np.zeros(2)
        for other in self.everyone:
            if other is person:
                continue
            away = np.array([person.x - other.x, person.y - other.y])
            dist = float(np.linalg.norm(away))
            gap = dist - (person.width + other.width) / 2
            if 1e-6 < dist and gap < PUSH_RANGE_M:
                v += away / dist * PUSH_SPEED * (PUSH_RANGE_M - gap) / PUSH_RANGE_M
        return v


def line_of_sight(plan, a, b, step_m: float = 0.1, over_low: bool = False) -> bool:
    """Whether nothing on the floor plan (wall or furniture) stands between points a and b.
    With `over_low` (a rider looking from eye height) low obstacles, such as coffee tables
    (`plan.low`, see webots/obstacles.py), do not count."""
    n = max(2, int(math.dist(a, b) / step_m))
    t = np.linspace(0.0, 1.0, n)[1:-1]
    xs, ys = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    rows, cols = plan.cells.shape
    c = np.clip(((xs + plan.width_m / 2) / plan.resolution).astype(int), 0, cols - 1)
    r = np.clip(((plan.height_m / 2 - ys) / plan.resolution).astype(int), 0, rows - 1)
    from envs.floorplan import WALL
    blocked = plan.cells[r, c] == WALL
    low = getattr(plan, "low", None)
    if over_low and low is not None:
        blocked &= ~low[r, c]
    return not bool(blocked.any())


def density_cap(cells: np.ndarray) -> int:
    """The most walking people for an area of floor (`cells`: route-grid cells): MAX_DENSITY per square
    metre, but at least 2."""
    return max(2, int(MAX_DENSITY * len(cells) * ROUTE_CELL_M ** 2))


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = np.concatenate([a, b]) if len(a) and len(b) else (a if len(a) else b)
    return np.unique(both, axis=0) if len(both) else both


def _route_ahead(route, here, length_m: float) -> list[tuple[float, float]]:
    """The part of the route from the point nearest `here` onwards, up to `length_m` long."""
    if not route:
        return [here]
    nearest = min(range(len(route)), key=lambda k: math.dist(route[k], here))
    out, total = [here], 0.0
    for a, b in zip(route[nearest:-1], route[nearest + 1:]):
        total += math.dist(a, b)
        out.append(tuple(b))
        if total >= length_m:
            break
    return out if len(out) > 1 else [here, tuple(route[-1])]


def _is_around(person, here, ahead) -> bool:
    """Whether a person is close to the chair, or to the route ahead of it."""
    return math.dist((person.x, person.y), here) <= ACTIVE_M or _near_route(person, ahead)


def _near_route(person, ahead) -> bool:
    p = (person.x, person.y)
    for a, b in zip(ahead[:-1], ahead[1:]):
        ab = (b[0] - a[0], b[1] - a[1])
        t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / max(ab[0] ** 2 + ab[1] ** 2, 1e-9)))
        if math.dist(p, (a[0] + t * ab[0], a[1] + t * ab[1])) <= ROUTE_CORRIDOR_M:
            return True
    return False


def _along(route, spacing):
    """Points every `spacing` metres along the polyline `route`."""
    out = [tuple(route[0])]
    for a, b in zip(route[:-1], route[1:]):
        n = max(1, int(math.dist(a, b) / spacing))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out
