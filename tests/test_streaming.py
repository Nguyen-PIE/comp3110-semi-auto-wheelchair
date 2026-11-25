"""Checks for the crowd (webots/people.py): Crowd.stream (people appear around the chair and along a
new route when the goal changes, out of its sight, and disappear when far away), and Crowd.make_way
(people standing in the chair's way step aside when it asks). No Webots needed: a stub scene stands
in for it.

    python tests/test_streaming.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import behaviours  # noqa: E402
from envs.floorplan import load_floorplan  # noqa: E402
from webots import people as P  # noqa: E402
from webots.people import Crowd, line_of_sight  # noqa: E402

PLAN = load_floorplan(str(ROOT / "third_party/floorplans/wheelchair-navi/carter_hospital_navigation.yaml"))
SLOTS = 40


class Chair:
    x = y = orientation = vel_x = vel_a = goal_x = goal_y = 0.0
    radius = 0.4


class Trip:
    attentive_share, elderly_share = 0.7, 0.0


class Scene:
    """Just what Crowd and envs/pedestrians.py read from the Webots scene."""
    TIMESTEP, HUMAN_DIAMETER = 0.1, 0.72
    MAX_ADVANCE_HUMAN = behaviours.settings()["speed_max"]
    static_interactions, moving_interactions, h_l_interactions = [], [], []
    crowd_style = "busy"
    trip = Trip()

    def __init__(self, seed=0, scale=1.5, plan=PLAN):
        self.plan, self.robot, self.goals, self._walkers, self.ticks = plan, Chair(), {}, {}, 0
        self.people_area = plan.spawn_cells
        self.crowd = Crowd(self)
        self.crowd.scale = scale
        self.rng = random.Random(seed)

    @property
    def dynamic_humans(self):
        return self.crowd.walking

    @property
    def static_humans(self):
        return self.crowd.standing

    def _free_slots(self):
        return max(0, SLOTS - len([p for p in self.crowd.everyone if p.kind == "person"]))

    def start(self, n=10):
        """A trip: the chair somewhere, a goal 10 to 25 m away, n people near the route."""
        rng, plan = self.rng, self.plan
        for _ in range(300):
            a, b = plan.random_point(rng), plan.random_point(rng)
            way = plan.route(a, b)
            if way and 12 < math.dist(a, b) < 22:
                break
        self.robot.x, self.robot.y, self.robot.goal_x, self.robot.goal_y = a[0], a[1], b[0], b[1]
        self.route = [a] + way
        self.people_area = plan.cells_near(self.route, 6.0)
        self.crowd.base_walkers = n
        n = min(round(n * self.crowd.scale), P.density_cap(self.people_area))   # as webots/bridge.py does
        self.crowd.reset(rng, n, 0, 0, self.people_area, [a, b])

    def set_goal(self, x, y):
        self.robot.goal_x, self.robot.goal_y = x, y
        here = (self.robot.x, self.robot.y)
        self.route = [here] + self.plan.route(here, (x, y))
        self.crowd.stream(force=True)

    def run(self, seconds):
        for _ in range(int(seconds / self.TIMESTEP)):
            self.ticks += 1
            self.crowd.step(self.TIMESTEP)
            if self.ticks % 10 == 0:
                self.crowd.stream()


def people_near(scene, points, within):
    return [p for p in scene.crowd.walking if min(math.dist((p.x, p.y), q) for q in points) < within]


def far_goal(scene, rng):
    """A new goal 15 to 30 m from the chair, well away from the old route."""
    here = (scene.robot.x, scene.robot.y)
    old = scene.route
    for _ in range(500):
        g = PLAN.random_point(rng)
        way = PLAN.route(here, g)
        if way and 15 < math.dist(here, g) < 30 and min(math.dist(g, q) for q in old) > 10:
            return g
    raise AssertionError("no far goal found")


def test_a_new_goal_gets_people_along_its_route_straight_away():
    scene = Scene(seed=3)
    scene.start(10)
    scene.run(5)
    before = len(people_near(scene, [(scene.robot.x, scene.robot.y)] + scene.route, 6.0))
    goal = far_goal(scene, random.Random(1))
    scene.set_goal(*goal)
    along = people_near(scene, scene.route, P.ROUTE_CORRIDOR_M + 1.0)
    assert len(along) >= 5, f"only {len(along)} people along the new route (there were {before} along the old one)"


def test_new_people_start_out_of_the_chairs_sight():
    scene = Scene(seed=4)
    scene.start(10)
    chair = (scene.robot.x, scene.robot.y)
    known = {id(p) for p in scene.crowd.walking}
    scene.set_goal(*far_goal(scene, random.Random(2)))
    new = [p for p in scene.crowd.walking if id(p) not in known]
    assert new, "nobody was added"
    for p in new:
        d = math.dist((p.x, p.y), chair)
        assert d >= P.SPAWN_MIN_M - 1e-6, f"appeared {d:.1f} m from the chair"
        assert d >= P.SEEN_M or not line_of_sight(PLAN, (p.x, p.y), chair) or d >= P.SPAWN_RELAXED_M, \
            f"appeared in plain sight {d:.1f} m away"


def test_people_far_from_the_chair_are_taken_away_and_never_more_than_the_world_can_draw():
    scene = Scene(seed=5)
    scene.start(12)
    scene.run(5)
    chair = (scene.robot.x, scene.robot.y)
    far = scene.crowd.walking[0]
    far.x, far.y = max(((x, y) for x, y in (PLAN.random_point(random.Random(k)) for k in range(400))),
                       key=lambda p: math.dist(p, chair))
    assert math.dist((far.x, far.y), chair) > P.DESPAWN_M
    far._traits.leader, far._traits.members = None, []      # on their own
    scene.crowd.stream()
    assert far not in scene.crowd.walking or line_of_sight(PLAN, (far.x, far.y), chair)
    scene.set_goal(*far_goal(scene, random.Random(3)))
    scene.run(30)
    assert len([p for p in scene.crowd.everyone if p.kind == "person"]) <= SLOTS


def test_a_group_is_taken_away_together():
    scene = Scene(seed=6)
    scene.start(14)
    leaders = [p for p in scene.crowd.walking if p._traits.members]
    if not leaders:
        return
    leader = leaders[0]
    chair = (scene.robot.x, scene.robot.y)
    spot = max((PLAN.random_point(random.Random(k)) for k in range(400)), key=lambda p: math.dist(p, chair))
    for q in [leader] + leader._traits.members:
        q.x, q.y = spot
    scene.crowd._take_away_far_people(chair)
    gone = [q for q in [leader] + leader._traits.members if q not in scene.crowd.walking]
    assert len(gone) in (0, 1 + len(leader._traits.members)), "part of a group was taken away"


def test_the_crowd_stays_about_the_same_size_while_the_chair_travels():
    scene = Scene(seed=7, scale=1.5)
    scene.start(10)
    counts = []
    for k in range(6):
        scene.set_goal(*far_goal(scene, random.Random(10 + k)))
        scene.robot.x, scene.robot.y = scene.robot.goal_x, scene.robot.goal_y    # the chair gets there
        scene.run(5)
        counts.append(len(people_near(scene, [(scene.robot.x, scene.robot.y)], P.ACTIVE_M)))
    assert min(counts) >= 3, f"people around the chair after each trip: {counts}"


def test_a_scale_of_one_keeps_about_the_usual_crowd_around_the_chair():
    scene = Scene(seed=8, scale=1.0)
    scene.start(6)
    here = (scene.robot.x, scene.robot.y)
    for _ in range(6):
        scene.run(10)
        ahead = P._route_ahead(scene.route, here, P.ROUTE_AHEAD_M)
        around = sum(1 for p in scene.crowd.walking if P._is_around(p, here, ahead))
        assert around <= 6 + 2, f"{around} people around the chair, the usual crowd is 6"


def test_a_narrow_corridor_stays_quiet():
    corridor = load_floorplan(str(ROOT / "webots/worlds/corridor_map.yaml"))
    for seed in (1, 2, 3):
        scene = Scene(seed=seed, scale=2.0, plan=corridor)
        scene.start(10)
        scene.run(40)
        here = (scene.robot.x, scene.robot.y)
        zone = scene.plan.cells_near([here], P.ACTIVE_M)
        near = len(people_near(scene, [here], P.ACTIVE_M))
        assert near <= P.density_cap(zone) + 2, \
            f"seed {seed}: {near} people within {P.ACTIVE_M} m in a corridor (cap {P.density_cap(zone)})"
        assert len(scene.crowd.walking) <= 8, f"seed {seed}: {len(scene.crowd.walking)} people in a corridor"


def test_people_who_arrive_walk_on_instead_of_turning_back():
    scene = Scene(seed=12)
    scene.start(8)
    person = scene.crowd.walking[0]
    forward = 0
    for _ in range(300):
        d = scene.crowd._onward(person, scene.people_area)
        v = (d.x - person.x, d.y - person.y)
        n = math.hypot(*v)
        forward += n > 0 and (v[0] * math.cos(person.orientation) + v[1] * math.sin(person.orientation)) / n > 0.3
    assert forward >= 270, f"only {forward} of 300 new destinations were ahead of them"


def test_streamed_people_walk_through_and_leave_instead_of_staying_around():
    scene = Scene(seed=13, scale=2.0)
    scene.start(8)
    seen, ids_over_time = set(), []
    for _ in range(120):
        scene.run(1)
        ids = {p.id for p in scene.crowd.walking if p._transit}
        seen |= ids
        ids_over_time.append(ids)
    assert seen, "nobody was streamed in"
    gone = [i for i in seen if i not in ids_over_time[-1]]
    assert len(gone) >= len(seen) // 2, f"only {len(gone)} of {len(seen)} streamed-in people ever left"


def test_a_walker_in_the_way_steps_aside_when_asked_and_walks_on_later():
    """A walker on the chair's route 2 m ahead: asked, they step off its way (where there is room,
    never into a wall), wait there while the chair is still close, and walk on after ASIDE_WAIT_S."""
    from webots.people import ASIDE_WAIT_S, CHAIR_RADIUS_M, Destination, Person, _line_gap
    scene = Scene(seed=3)
    scene.start(n=0)
    route = scene.route
    k = next(i for i in range(1, len(route)) if math.dist(route[0], route[i]) > 2.0)
    x, y = route[k]
    walker = Person(900, x, y, goal_x=route[0][0], goal_y=route[0][1])
    scene.crowd.walking.append(walker)
    scene.goals[walker.id] = Destination(*route[0])
    chair = (scene.robot.x, scene.robot.y)
    assert scene.crowd.make_way(x + 20.0, y + 20.0, chair, route) is None
    assert scene.crowd.make_way(x, y, chair, route) is True
    scene.run(4.0)
    assert walker._aside is not None, "stopped waiting while the chair was still right there"
    assert _line_gap((walker.x, walker.y), route) > CHAIR_RADIUS_M + walker.width / 2, "still in the chair's way"
    assert PLAN.clearance_at(walker.x, walker.y) > walker.width / 2 - 0.05, "stepped into a wall"
    scene.run(ASIDE_WAIT_S)
    assert walker._aside is None, "still waiting at the side"


def test_a_walker_asked_to_make_way_never_heads_through_someone_and_walks_on_if_kept_from_their_spot():
    """The spot a walker is sent to is one they can walk straight to without brushing past anyone.
    If someone steps into their way anyway, they walk on after ASIDE_STUCK_S instead of pressing on."""
    from webots.people import ASIDE_STUCK_S, Destination, Person, _segment_gap
    scene = Scene(seed=3)
    scene.start(n=0)
    route = scene.route
    k = next(i for i in range(1, len(route)) if math.dist(route[0], route[i]) > 2.0)
    x, y = route[k]
    walker = Person(900, x, y, goal_x=route[0][0], goal_y=route[0][1])
    scene.crowd.walking.append(walker)
    scene.goals[walker.id] = Destination(*route[0])
    near = [Person(901 + i, x + 0.9 * math.cos(a), y + 0.9 * math.sin(a), type="static")
            for i, a in enumerate((0.5, 2.6, 4.4)) if PLAN.clearance_at(x + 0.9 * math.cos(a), y + 0.9 * math.sin(a)) > 0.4]
    scene.crowd.standing += near
    chair = (scene.robot.x, scene.robot.y)
    if scene.crowd.make_way(x, y, chair, route):
        q = walker._aside
        for o in near:
            assert _segment_gap((o.x, o.y), (x, y), q) >= min(math.dist((o.x, o.y), (x, y)), 0.82) - 1e-6, \
                "sent past someone standing close"
    walker._aside, walker._aside_left = None, 0.0
    scene.crowd.standing = [p for p in scene.crowd.standing if p not in near]
    assert scene.crowd.make_way(x, y, chair, route) is True
    gx, gy = walker._aside
    d = math.dist((x, y), (gx, gy))
    blocker = Person(950, x + (gx - x) / d * 0.75, y + (gy - y) / d * 0.75, type="static")   # steps right into their way
    scene.crowd.standing.append(blocker)
    scene.run(ASIDE_STUCK_S + 1.0)
    assert walker._aside is None, "kept pressing towards a spot someone is standing in the way of"


def test_someone_pressed_against_the_chair_takes_another_way_round_it():
    """A walker whose next step would go into the chair stays put that step, as in 2D, and plans
    another way round it rather than pressing against it until they give up after 3 s. So does one
    creeping along a wall or a bench into the chair."""
    from webots.people import Destination, Person
    scene = Scene(seed=3)
    scene.start(n=0)
    r = scene.robot
    a = math.atan2(r.goal_y - r.y, r.goal_x - r.x)
    x, y = r.x + 0.77 * math.cos(a), r.y + 0.77 * math.sin(a)        # 1 cm in front of the chair
    goal = (r.x - 3.0 * math.cos(a), r.y - 3.0 * math.sin(a))        # straight through it
    walker = Person(900, x, y, goal_x=goal[0], goal_y=goal[1], avoids_robot=False)
    walker._last_velocity = [-math.cos(a) * 1.0, -math.sin(a) * 1.0]
    scene.crowd.walking.append(walker)
    scene.goals[walker.id] = Destination(*goal)
    scene.ticks += 1
    scene.crowd.step(scene.TIMESTEP)
    assert math.dist((walker.x, walker.y), (r.x, r.y)) >= r.radius + walker.width / 2, "stepped into the chair"
    assert walker._replan, "pressed against it, no other way"
    walker._replan, walker._last_velocity = False, [-math.cos(a) * 0.1, -math.sin(a) * 0.1]
    walker.x, walker.y = r.x + 0.84 * math.cos(a), r.y + 0.84 * math.sin(a)        # 8 cm away, creeping closer
    walker.goal_x, walker.goal_y = r.x, r.y
    scene.ticks += 1
    scene.crowd.step(scene.TIMESTEP)
    assert walker._replan, "creeping into the chair, no other way"


def test_someone_walking_straight_at_the_chair_stops_just_short_of_it():
    """Walking up to the chair, people (and forklifts) stop STOP_SHORT_M short of it rather than right
    against it, where the smallest move of the chair would touch them."""
    from webots.people import STOP_SHORT_M, Destination, Person
    scene = Scene(seed=3)
    scene.start(n=0)
    r = scene.robot
    a = math.atan2(r.goal_y - r.y, r.goal_x - r.x)
    goal = (r.x - 3.0 * math.cos(a), r.y - 3.0 * math.sin(a))
    for kind, width in (("person", 0.72), ("forklift", 1.8)):
        d = r.radius + width / 2 + 0.3
        walker = Person(900, r.x + d * math.cos(a), r.y + d * math.sin(a), goal_x=goal[0], goal_y=goal[1],
                        kind=kind, width=width, avoids_robot=False)
        scene.crowd.walking = [walker]
        scene.goals[walker.id] = Destination(*goal)
        for _ in range(30):
            scene.ticks += 1
            scene.crowd.step(scene.TIMESTEP)
            gap = math.dist((walker.x, walker.y), (r.x, r.y)) - r.radius - width / 2
            assert gap >= STOP_SHORT_M - 1e-9, f"a {kind} walked up to {gap:.3f} m from the chair"


def test_someone_walked_from_a_phone_bumps_into_the_chair():
    """A person walked from a phone into the chair bumps into it (a hit, as in real life) but does not
    walk through it. crossing.py's scripted crosser, also a player, is still stopped right at it."""
    from webots.bridge import CHAIR_RADIUS_M, PERSON_DIAMETER_M
    from webots.people import Person
    scene = Scene(seed=3)
    scene.start(n=0)
    r = scene.robot
    for by_phone in (True, False):
        walker = Person(900, r.x + 1.5, r.y, player=1, by_phone=by_phone, orientation=math.pi)
        walker._player_move = (1.3, 0.0)                     # straight at the chair, full speed
        scene.crowd.walking = [walker]
        for _ in range(20):
            scene.ticks += 1
            scene.crowd.step(scene.TIMESTEP)
        d = math.dist((walker.x, walker.y), (r.x, r.y))
        assert walker.x > r.x, "walked through the chair"
        if by_phone:
            assert d < CHAIR_RADIUS_M + PERSON_DIAMETER_M / 2, f"stopped {d:.3f} m away: no hit"
        else:
            assert d >= CHAIR_RADIUS_M + walker.width / 2 - 1e-9, "the scripted crosser walked into it"


def test_people_standing_to_talk_or_wait_are_never_asked():
    from webots.people import Person
    scene = Scene(seed=3)
    scene.start(n=0)
    x, y = scene.route[2]
    scene.crowd.standing += [Person(900, x, y, type="static"), Person(901, x + 0.7, y, type="static")]
    assert not scene.crowd.askable(x, y)
    assert scene.crowd.make_way(x, y, (scene.robot.x, scene.robot.y), scene.route) is None


def test_someone_standing_in_the_middle_of_a_corridor_closes_it_but_not_in_a_hall():
    """What Crowd.reset checks before letting people stand somewhere: they must leave the chair a way."""
    from webots.people import _closes_a_way
    rng = random.Random(5)

    def spot(lo, hi, straight):
        for _ in range(400000):
            x, y = rng.uniform(-PLAN.width_m / 2, PLAN.width_m / 2), rng.uniform(-PLAN.height_m / 2, PLAN.height_m / 2)
            c = PLAN.clearance_at(x, y)
            if not lo <= c <= hi:
                continue
            if not straight:
                return x, y
            for k in range(8):   # a corridor: as narrow 1.5 m either way along it, walls on both sides across it
                a = k * math.pi / 8
                along = [PLAN.clearance_at(x + d * math.cos(a), y + d * math.sin(a)) for d in (-1.5, 1.5)]
                across = [PLAN.clearance_at(x - d * math.sin(a), y + d * math.cos(a)) for d in (-c - 0.1, c + 0.1)]
                if all(abs(v - c) < 0.1 for v in along) and all(v < 0.15 for v in across):
                    return x, y
        raise AssertionError("no such spot")

    assert _closes_a_way(PLAN, [spot(0.95, 1.05, straight=True)], [])
    assert not any(_closes_a_way(PLAN, [spot(3.0, 9.0, straight=False)], []) for _ in range(10))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "-", repr(exc))
    sys.exit(1 if failed else 0)
