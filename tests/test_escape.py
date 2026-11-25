"""Checks for the way out of a gap in assist/controller.py: when the safety stop has held the chair
between people for a while, it finds open space and drives out. No simulator needed.

    python tests/test_escape.py          (or: python -m pytest tests, if pytest is installed)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assist.controller import (ASK_EVERY_STEPS, ASK_HELD_STEPS, BLOCKER_KEEP_STEPS, BLOCKER_STEPS,  # noqa: E402
                               ESCAPE_ROOM_M, N_RAYS, PASSING_GAP_M, STUCK_STEPS, AssistController)
from assist.sensors import MAX_RANGE_M  # noqa: E402


def person(x, y, radius=0.36, vx=0.0, vy=0.0):
    """A tracked person seen from a chair (radius 0.4) at the origin."""
    v2 = vx * vx + vy * vy
    return SimpleNamespace(x=x, y=y, radius=radius, id=id((x, y)), vx=vx, vy=vy, speed=math.hypot(vx, vy),
                           distance=math.hypot(x, y) - 0.4 - radius,
                           t_closest=max(0.0, -(x * vx + y * vy) / v2) if v2 > 1e-9 else 0.0)


def controller(people, walls=None, heading=0.0):
    """A controller for a chair at the origin facing `heading`, with these people tracked and the wall
    distances `walls` (a function of the world-frame direction), without a simulator."""
    c = AssistController.__new__(AssistController)
    chair = SimpleNamespace(x=0.0, y=0.0, orientation=heading, radius=0.4, vel_x=0.0, vel_a=0.0)
    c.env = SimpleNamespace(unwrapped=SimpleNamespace(robot=chair))
    c.tracker = SimpleNamespace(tracks=people)
    c.still = set()
    c.ranges = np.array([walls(heading + 2 * math.pi * k / N_RAYS) if walls else MAX_RANGE_M
                         for k in range(N_RAYS)], dtype=np.float32)
    c.target_local = (1.0, 0.0, 5.0)            # the route goes straight ahead
    c.blocked_steps = c.escape_left = 0
    c.speed, c.safety_stop = 0.0, False
    c.escape_heading, c.escape_from = 0.0, (0.0, 0.0)
    c.base = SimpleNamespace(steer=lambda x, y, dist, speed: np.array([speed, 0.0, math.atan2(y, x)], dtype=np.float32))
    return c


def wall_to_the_south(direction):
    """A wall 0.6 m south of the chair: distance to it in a world-frame direction."""
    south = -math.sin(direction)
    return 0.6 / south if south > 1e-6 else MAX_RANGE_M


def test_finds_the_open_space_above_when_boxed_in_below():
    people = [person(0.55, 0.0), person(0.0, -0.8), person(0.6, 0.9)]     # ahead, below, ahead-and-above
    c = controller(people, wall_to_the_south)
    h = c.open_heading()
    assert h is not None, "found no way out although there is open space"
    assert c.clearance_along(h) >= ESCAPE_ROOM_M
    assert math.sin(h) > 0 or math.cos(h) < 0, "headed towards the wall or the people"


def test_a_ring_of_people_gives_no_way_out():
    ring = [person(1.0 * math.cos(a), 1.0 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
    assert controller(ring).open_heading() is None


def test_does_not_drive_through_a_person():
    c = controller([person(0.9, 0.0)])
    assert c.clearance_along(0.0) < 0.3                  # straight at them
    assert c.clearance_along(math.pi) > ESCAPE_ROOM_M    # away from them


def test_backs_off_only_when_every_way_on_is_closed():
    """1.2 m of room ahead-right, much more behind: it gets out ahead-right, not back the way it came."""
    people = [person(1.0, 0.3), person(0.0, 0.9), person(-0.2, -0.95), person(1.4, -1.4)]
    c = controller(people)
    h = c.open_heading()
    assert h is not None and math.cos(h) > 0, "backed off although a way on was open"
    c.tracker.tracks = people + [person(0.9, -0.75)]   # that way closes too: back off
    h = c.open_heading()
    assert h is not None and math.cos(h) < 0


def test_prefers_the_way_to_the_goal_when_several_are_open():
    c = controller([person(0.0, -0.8)])
    h = c.open_heading()
    assert abs(h) < 0.5, "turned away from the route although it was open"


def test_starts_after_being_stopped_for_two_seconds_and_ends_after_a_metre():
    c = controller([person(0.5, 0.0), person(0.0, -0.8)], wall_to_the_south)
    chair = c.env.unwrapped.robot
    for _ in range(STUCK_STEPS - 1):
        c.safety_stop = True
        assert c.escape_command() is None
    c.safety_stop = True
    command = c.escape_command()
    assert command is not None, "did not start looking for a way out"
    assert command[0] >= 0 and c.safety_stop is False
    chair.x = 1.2                                           # has got clear of the gap
    assert c.escape_command() is None and c.escape_left == 0


def test_getting_out_never_drives_into_someone():
    someone = person(0.85, 0.0)                  # 9 cm in front of the chair
    c = controller([someone])
    chair = c.env.unwrapped.robot
    for _ in range(STUCK_STEPS):
        c.safety_stop = True
        command = c.escape_command()
    assert command is not None, "did not start looking for a way out"
    for _ in range(20):                          # even if it never turned, it stops short of them
        chair.x += float(command[0]) * 1.2 * 0.1
        gap = math.hypot(someone.x - chair.x, someone.y - chair.y) - chair.radius - someone.radius
        assert gap >= PASSING_GAP_M - 1e-6, f"drove to {gap:.3f} m from the person"
        command = c.escape_command()
        assert command is not None
    chair.orientation = c.escape_heading         # has turned towards the way out
    command = c.escape_command()
    assert command is not None and command[0] > 0, "did not drive out once facing open space"


def test_getting_out_never_steps_into_someone_walking_past():
    """The way out is open now, but someone walking across it would be there in a moment: wait."""
    c = controller([person(0.0, 2.0)], walls=wall_to_the_south)   # boxed in below: the way out is north
    for _ in range(STUCK_STEPS):
        c.safety_stop = True
        command = c.escape_command()
    chair = c.env.unwrapped.robot
    chair.orientation = c.escape_heading
    c.tracker.tracks = [person(-0.7, 0.8, vx=1.2)]   # walking east, about to cross just in front
    command = c.escape_command()
    assert command is not None and command[0] == 0.0, "drove into the path of someone walking past"
    c.tracker.tracks = [person(-0.7, 0.8, vx=-1.2)]  # walking away west: go
    command = c.escape_command()
    assert command is not None and command[0] > 0, "waited for someone walking away"


def test_held_by_a_wall_corner_it_gets_out_the_same_way():
    """The route runs into a wall corner and the wall brake holds the chair still, nobody near:
    after the same 2 s it drives out to open space, as from a crowd."""
    c = controller([], walls=wall_to_the_south)
    c.base.wall_braked = True
    for _ in range(STUCK_STEPS):
        c.safety_stop = False
        command = c.escape_command()
    assert command is not None, "stayed stuck against the wall"
    assert math.sin(c.escape_heading) > 0, "the way out should lead away from the wall"


def test_the_way_out_is_never_one_the_wall_brake_will_not_drive():
    """The lidar sees open floor straight ahead, but the floor plan has a shelf corner 20 cm ahead-left
    that the wall brake will not pass: the way out leads off to the right of it instead."""
    def clearance_at(x, y):                    # a corner at (0.3, 0.42): 1 cm to spare passing it straight on
        return min(math.dist((x, y), (0.3, 0.42)), 0.6 - y if y < 0.6 else 0.0, 3.0)

    c = controller([])
    c.base.plan = SimpleNamespace(clearance_at=clearance_at)
    c.base.wall_braked = True
    for _ in range(STUCK_STEPS):
        c.safety_stop = False
        command = c.escape_command()
    assert command is not None, "stayed held by the wall brake"
    assert math.sin(c.escape_heading) < -0.1, "headed straight on past the corner the brake holds it at"
    assert c.clearance_along(0.0) < 0.3


def test_keeps_its_way_while_nobody_stands_on_it():
    """Someone pausing on the route for a moment does not send the chair another way (in two-way
    traffic it turned back and forth between the two sides of a row of benches); someone who has
    stood on it for a while (in `fixed`) does."""
    import random
    from baselines.route_follower import REPLAN_EVERY, RouteFollower
    from envs.floorplan import load_floorplan, route_length
    plan = load_floorplan(str(ROOT / "third_party/floorplans/wheelchair-navi/carter_hospital_navigation.yaml"))
    rng = random.Random(4)
    while True:
        a, b = plan.random_point(rng), plan.random_point(rng)
        way = plan.route(a, b)
        if way and 10 < math.dist(a, b) < 20:
            break
    chair = SimpleNamespace(x=a[0], y=a[1], goal_x=b[0], goal_y=b[1], orientation=0.0, radius=0.4)
    env = SimpleNamespace(unwrapped=SimpleNamespace(robot=chair, plan=plan))
    for keep in (True, False):
        rf = RouteFollower(env)
        rf.keep_open_route = keep
        rf.reset()
        rf.target()
        first = rf.path
        ahead = rf.path[2] if len(rf.path) > 2 else rf.path[1]
        rf.avoid, rf.steps = [(ahead[0], ahead[1], 0.46)], REPLAN_EVERY
        rf.target()
        assert (rf.path is first) == keep, "changed its way for someone pausing" if keep else "old behaviour changed"
    rf.keep_open_route = True
    rf.fixed, rf.steps = [(ahead[0], ahead[1], 0.46)], 2 * REPLAN_EVERY
    rf.target()
    assert rf.path is not first and rf._route_open((chair.x, chair.y)), "kept a way someone has stood on for a while"
    rf.path, rf.progress = [(chair.x, chair.y), (chair.x + 8.0, chair.y), b], 0    # a long way round ...
    rf.avoid, rf.fixed, rf.steps = [], [], 3 * REPLAN_EVERY                        # ... and the short way opens
    rf.target()
    assert route_length((chair.x, chair.y), rf.route) < math.dist(a, b) + 3.0, "kept a long way round"


def test_someone_who_keeps_the_chair_held_up_is_planned_round():
    """A walker shuffling about right in front (never still long enough to count as standing) has
    held the chair up for 4 s: the route goes round them for a while instead of both waiting."""
    shuffler = person(0.9, 0.0, vx=0.3)
    c = controller([shuffler])
    c.steps, c.blockers, c.made_way, c.base.fixed, c.base.path = 0, {}, {}, [], ["old route"]
    c.base.replan_wanted = False
    c.held_steps = BLOCKER_STEPS - 1
    c._blockers([shuffler])
    assert c.base.fixed == [], "not held up long enough yet"
    c.held_steps, c.made_way = BLOCKER_STEPS, {shuffler.id: 0}     # just asked to make way: stepping aside
    c._blockers([shuffler])
    assert c.base.fixed == [] and not c.base.replan_wanted, "planned round someone stepping aside as asked"
    c.made_way = {}
    c.held_steps = BLOCKER_STEPS
    c._blockers([shuffler])
    assert c.base.fixed and c.base.replan_wanted, "should plan round the person holding it up"
    c.steps += BLOCKER_KEEP_STEPS + 1
    c.held_steps, c.base.fixed = 0, []
    c._blockers([shuffler])
    assert c.base.fixed == [], "kept planning round them for good"


def test_a_faster_walker_right_beside_is_let_past():
    c = controller([person(0.0, 0.95, vx=1.4)])      # 19 cm from the chair's left side, overtaking
    c.env.unwrapped.robot.vel_x = 0.4
    assert c._overtaken()
    c.tracker.tracks = [person(0.0, 0.95, vx=-1.4)]   # coming the other way: passes by itself
    assert not c._overtaken()
    c.tracker.tracks = [person(0.0, 1.5, vx=1.4)]     # 74 cm away: room enough
    assert not c._overtaken()


def test_a_walker_holding_the_chair_up_is_asked_to_make_way_but_people_standing_are_not():
    """Only walkers are asked ("Excuse me"), once the chair has been held up 3 s, and again every 3 s.
    People standing to talk or wait (the scene says they are not askable) are gone round instead."""
    walker = person(0.9, 0.0, vx=0.2)
    c = controller([walker])
    asked = []
    c.env.unwrapped.askable = lambda x, y: (x, y) == (walker.x, walker.y)
    c.env.unwrapped.make_way = lambda x, y, lane: asked.append((x, y)) or True
    c.env.unwrapped.robot.goal_x, c.env.unwrapped.robot.goal_y = 10.0, 0.0
    c.base.plan = SimpleNamespace(route=lambda a, b: [b])
    c.steps, c.asked_at, c.stalled_steps, c.base.path, c.made_way = 0, -ASK_EVERY_STEPS, ASK_HELD_STEPS - 1, ["old route"], {}
    c.base.replan_wanted = False
    c._ask((0.0, 0.0))
    assert not c.asking and not asked, "asked before being held up 3 s"
    c.stalled_steps = ASK_HELD_STEPS
    c._ask((0.0, 0.0))
    assert c.asking and asked == [(0.9, 0.0)] and c.base.replan_wanted
    assert walker.id in c.made_way, "not noted as stepping aside"
    c.steps += 1
    c._ask((0.0, 0.0))
    assert asked == [(0.9, 0.0)], "asked again at once"
    c.steps += ASK_EVERY_STEPS
    c._ask((0.0, 0.0))
    assert len(asked) == 2, "did not ask again after 3 s"
    standing = person(0.9, 0.0)
    c = controller([standing])
    c.env.unwrapped.askable = lambda x, y: False
    c.env.unwrapped.make_way = lambda x, y, lane: asked.append((x, y)) or True
    c.steps, c.asked_at, c.stalled_steps, c.made_way = 0, -ASK_EVERY_STEPS, ASK_HELD_STEPS, {}
    c._ask((0.0, 0.0))
    assert not c.asking and len(asked) == 2, "asked someone standing to move"


def test_driving_on_waits_for_someone_walking_right_across_its_way():
    """The safety stop looks at where people are now; someone walking fast across the chair's way
    a few cm ahead would be hit by the time it reacts. Driving on, the chair waits for them."""
    crossing = person(0.95, -0.35, vy=1.2)        # 23 cm ahead-right, walking across to the left
    c = controller([crossing])
    c.env.unwrapped.robot.vel_x = 0.6
    c.engaged, c.offset, c.speed, c.last_action = False, 0.0, 0.5, np.zeros(2)
    c.needs_adjusting = lambda: False
    c.escape_command = lambda: None
    c._overtaken = lambda: False
    c.room_to_steer = lambda offset: 1.0
    command = c.command(np.zeros(2))
    assert c.safety_stop and command[0] == 0.0, "drove on into someone walking across"
    c.tracker.tracks = [person(0.95, -1.4, vy=-1.2)]   # walking away from its way: drives on
    c.speed = 0.5
    command = c.command(np.zeros(2))
    assert not c.safety_stop and command[0] > 0.0


def test_waits_for_someone_walking_in_towards_its_front_half():
    """A distracted walker 18 cm from its front-left corner, walking in towards it: the chair does not
    set off (they could curve into its side). Walking away, or behind it, they are no reason to wait."""
    c = controller([person(0.3, 0.88, vx=-0.6, vy=-0.6)])
    assert c.someone_in_the_way()
    c.tracker.tracks = [person(0.3, 0.88, vx=0.6, vy=0.6)]
    assert not c.someone_in_the_way()
    c.tracker.tracks = [person(-0.3, 0.88, vx=0.6, vy=-0.6)]
    assert not c.someone_in_the_way()


def test_half_speed_with_someone_walking_close_in_front_or_beside():
    c = controller([person(1.1, 0.3, vx=-1.0)])     # 38 cm ahead-left, walking towards it
    assert c._crowded()
    c.tracker.tracks = [person(-1.1, 0.0, vx=1.0)]   # just behind: not its concern
    assert not c._crowded()
    c.tracker.tracks = [person(1.1, 0.3)]           # standing still: the tracker has them right
    assert not c._crowded()
    c.tracker.tracks = [person(1.2, 0.0, vx=-1.0)]  # 44 cm ahead is close too, 64 cm is not
    assert c._crowded()
    c.tracker.tracks = [person(1.4, 0.0, vx=-1.0)]
    assert not c._crowded()


def test_out_of_the_gap_it_plans_again_from_where_it_is():
    """Once it has driven out, the route follower plans from where the chair is, so it does not
    chase a point on its old route behind it."""
    c = controller([person(0.5, 0.0), person(0.0, -0.8)], wall_to_the_south)
    c.base.replan_from_here = False
    for _ in range(STUCK_STEPS):
        c.safety_stop = True
        command = c.escape_command()
    assert command is not None and not c.base.replan_from_here
    c.env.unwrapped.robot.x = 1.2
    assert c.escape_command() is None and c.base.replan_from_here


def test_gets_out_close_to_the_way_of_the_route():
    """Two ways out within a quarter turn of the route: straight on with 1.2 m of room, or about 70
    degrees off with 2 m or more. It takes the one along the route (the other left the route behind
    it, and it turned round to get back to it)."""
    def walls(direction):
        d = math.degrees((direction + math.pi) % (2 * math.pi) - math.pi)
        return 1.6 if abs(d) < 33.75 else 3.0 if 33.75 <= d < 101.25 else 0.5
    c = controller([], walls)
    h = c.open_heading()
    assert h is not None and abs(h) < 0.3, f"went {math.degrees(h):.0f} degrees off the route"


def _u_turn_follower():
    """A route follower (as the assistant uses it) for a chair at the origin facing east, the goal 10 m
    east. Someone standing at (2, 0) blocks the way on; going round them means going back west."""
    from baselines.route_follower import RouteFollower
    detour = [(-3.0, 0.0), (-3.0, 5.0), (10.0, 5.0), (10.0, 0.0)]
    plan = SimpleNamespace(route=lambda a, b, around=None: list(detour) if around else [b],
                           clearance_at=lambda x, y: 3.0)
    chair = SimpleNamespace(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0, orientation=0.0, radius=0.4, vel_x=0.0)
    rf = RouteFollower(SimpleNamespace(unwrapped=SimpleNamespace(robot=chair, plan=plan)))
    rf.keep_open_route = True
    rf.reset()
    rf.target()
    return rf, detour


def test_waits_before_turning_back_the_way_it_came():
    """Someone who has stood on its way for 5 s, where the only other way goes back the way it came:
    it keeps its way and waits UTURN_WAIT_STEPS held up before turning round; if they move on first,
    it never turns round."""
    from baselines.route_follower import REPLAN_EVERY, UTURN_WAIT_STEPS
    rf, detour = _u_turn_follower()
    first = rf.path
    rf.fixed = rf.avoid = [(2.0, 0.0, 0.46)]
    rf.steps = REPLAN_EVERY
    for _ in range(UTURN_WAIT_STEPS // 2):
        rf.target()
    assert rf.path is first and rf.uturn_waiting, "turned back at once"
    rf.fixed = rf.avoid = []                        # they walk on: the way on is open again
    for _ in range(UTURN_WAIT_STEPS):
        rf.target()
    assert rf.path is first and not rf.uturn_waiting, "turned back although the way on opened"
    rf.fixed = rf.avoid = [(2.0, 0.0, 0.46)]        # they stand there again, and stay
    rf.replan_wanted = True
    for _ in range(UTURN_WAIT_STEPS + 20):
        rf.target()
    assert rf.path[1:] == detour, "never took the way back round them"


def test_once_turned_back_it_keeps_its_new_way_for_a_while():
    from baselines.route_follower import COMMIT_STEPS, UTURN_WAIT_STEPS
    rf, detour = _u_turn_follower()
    rf.fixed = rf.avoid = [(2.0, 0.0, 0.46)]
    rf.replan_wanted = True
    for _ in range(UTURN_WAIT_STEPS + 20):
        rf.target()
    turned = rf.path
    assert turned[1:] == detour
    rf.fixed = rf.avoid = []                        # they walk off: the old way is open and much shorter
    rf.robot = rf.env.unwrapped.robot
    rf.robot.orientation = math.pi                  # (now facing west, along the new way)
    for _ in range(COMMIT_STEPS // 2):
        rf.replan_wanted = True
        rf.target()
    assert rf.path is turned, "swung straight back round"
    for _ in range(COMMIT_STEPS):
        rf.replan_wanted = True
        rf.target()
    assert rf.path is not turned, "kept the long way round for good"


def test_not_stuck_means_no_escape():
    c = controller([person(3.0, 0.0)])
    for _ in range(100):
        c.safety_stop = False
        assert c.escape_command() is None


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
