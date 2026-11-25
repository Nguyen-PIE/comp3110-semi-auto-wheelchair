"""The bridge between Webots and the 2D simulator's code.

The assistant (assist/), the drivers (baselines/) and the training code (training/)
never talk to the simulator directly. They only read a handful of things from
`env.unwrapped`: where the chair is and how fast it is moving, where its goal is,
where the people are, the floor plan, and a few limits (time step, chair radius,
top speed). WebotsScene provides exactly those, filled in from Webots, so all of that
code runs unchanged in 3D, including the policy trained in 2D.

What comes from where:

    chair position and speed     Webots (the supervisor reads the chair's true pose:
                                 perfect localisation, as in the 2D simulator)
    walls around the chair       the chair's lidar in Webots (16 of its 360 rays), or
                                 the floor plan with walls="map", exactly as in 2D
    people                       their true positions plus 3 cm of noise, read by the
                                 assistant's own tracker, exactly as in 2D
    route to the goal            planned on the 2D floor plan (a real chair would also
                                 navigate with a map of the building); open_world.py gives
                                 the chair only its own map, from its lidar (explore.py)
    collisions                   same rules as 2D: the chair's centre closer than 0.4 m
                                 to a wall, or closer than 0.76 m to a person's centre

The chair is driven like the real one: the command (forward speed, turn rate) is
turned into a speed for each of the two drive wheels, and Webots' physics does the rest.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
sys.path.insert(0, str(ROOT))

from webots import launch  # noqa: E402  (finds Webots' Python library)

launch.add_controller_library_to_path()
from controller import Keyboard, Supervisor  # noqa: E402

import assist.controller  # noqa: E402
from assist.sensors import MAX_RANGE_M, N_RAYS  # noqa: E402
from envs import behaviours  # noqa: E402
from envs.floorplan import load_floorplan, route_length  # noqa: E402
from webots.people import STREAM_EVERY_S, Crowd, density_cap  # noqa: E402

STEP_S = 0.1                 # one decision every 0.1 s, as in the 2D simulator
CHAIR_RADIUS_M = 0.4
PERSON_DIAMETER_M = 0.72
MAX_SPEED = 1.2              # m/s
MAX_TURN = 0.8               # rad/s
WHEEL_RADIUS_M = 0.15        # must match protos/Wheelchair.proto
WHEEL_GAP_M = 0.5            # distance between the two drive wheels
PEDESTRIAN_HEIGHT = 1.27
PARKED = (0.0, -1000.0)
TRAIL_HEIGHT_M = 0.02        # just above the floor
TRAIL_NODE = ("DEF TRAIL_POSE Pose { children [ "   # in a Pose: a bare Shape cannot be hidden from cameras
              "DEF TRAIL Shape { appearance Appearance { material Material { diffuseColor 0 0 0 "
              "emissiveColor 0.1 0.45 1 } } geometry IndexedLineSet { coord Coordinate { point [ 0 0 -1, 0 0 -1 ] } "
              "coordIndex [ 0 1 ] } castShadows FALSE isPickable FALSE } ] }")   # a line needs two points
GAIT_CYCLE_M = 0.22 * 8      # metres walked per full walking-animation cycle (Webots' own pedestrian.py)
# The rider: a Pedestrian model (make_world.py RIDER_NODE) sitting in the chair, moved with it every step.
RIDER_SEAT = (-0.08, 0.0, 0.90)   # where the model's origin goes, in the chair's frame, so the hips are on the seat
RIDER_POSE = {"leftLegAngle": -1.52, "rightLegAngle": -1.52, "leftLowerLegAngle": 1.45, "rightLowerLegAngle": 1.45,
              "leftFootAngle": 0.05, "rightFootAngle": 0.05, "leftArmAngle": -0.30, "rightArmAngle": -0.30,
              "leftLowerArmAngle": -1.25, "rightLowerArmAngle": -1.25, "headAngle": 0.05}   # sitting, hands forward

# The walking animation of Webots' Pedestrian model (from its sample controller): 8 key poses.
JOINTS = ["leftArmAngle", "leftLowerArmAngle", "leftHandAngle", "rightArmAngle", "rightLowerArmAngle",
          "rightHandAngle", "leftLegAngle", "leftLowerLegAngle", "leftFootAngle", "rightLegAngle",
          "rightLowerLegAngle", "rightFootAngle", "headAngle"]
GAIT = np.array([
    [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74], [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
    [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0], [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
    [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38], [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
    [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4], [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
    [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07], [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
    [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49], [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
    [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]])
GAIT_HEIGHT = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
STAND = np.zeros(len(JOINTS))   # the Pedestrian model's own rest pose: upright, feet together, arms down
STAND_BLEND_S = 0.4             # a walker who stops settles into standing over this long, and back again
STILL_SPEED = 0.05              # m/s: slower than this counts as standing still


CROWDS = ("busy", "spread", "events")
# What can happen in each building, besides older people and people stepping out of rooms (everywhere).
BUILDING_EVENTS = {
    "corridor": ("two_way_traffic",),
    "fp_hospital": ("waiting_area", "slow_patients", "emergency_trolley"),
    "fp_office": ("meeting_ends", "phone_walker"),
    **{b: ("box_carriers", "pallet", "forklift") for b in ("fp_warehouse", "fp_warehouse2", "fp_warehouse3", "fp_depot")},
    "fp_house": ("child", "pet"),
}
EVENT_PEOPLE = {"meeting_ends": 8, "waiting_area": 6, "emergency_trolley": 2}   # pedestrian models an event may need
EVENT_OBJECTS = {"emergency_trolley": "trolley", "forklift": "forklift", "pallet": "pallet", "child": "child",
                 "pet": "pet"}
EVENTS = dict(elderly_share=0.3, room_exits=2)   # the "events" test crowd: spread out, plus these and
                                                 # every event of the building


@dataclass
class Variety:
    """How one training trip differs from the standard one (with variety on: webots_env.py, for
    training only; testing always uses the standard trip, so results stay comparable).
    The defaults are the standard trip."""
    crowd_scale: float = 1.0       # walking people: this many times the scenario's usual number
    attentive_share: float = 0.7   # share of people who make room for the chair (the rest are distracted)
    people_within: float | None = 6.0  # people start within this many metres of the chair's route
                                       # (None: anywhere in the building)
    speed_gain: float = 1.0        # how much of the commanded speed the chair actually gives
    turn_gain: float = 1.0         # ... and of the commanded turn (load, grip, motor differences)
    delay: bool = False            # the chair reacts one step (0.1 s) late
    lidar_noise_m: float = 0.0     # noise on every lidar distance (standard deviation)
    lidar_dropout: float = 0.0     # share of lidar rays that miss and read "nothing there"
    elderly_share: float = 0.0     # share of walking people who are older (people.py: slow, stop, notice late)
    room_exits: int = 0            # people who step out of a room across the route as the chair comes
    events: tuple = ()             # the building's own events in this trip (BUILDING_EVENTS)

    @classmethod
    def draw(cls, rng: random.Random, spread_share: float = 0.5, building_events: tuple = ()) -> "Variety":
        """spread_share of trips get people anywhere in the building, the rest people near the route."""
        return cls(crowd_scale=rng.uniform(0.6, 1.5), attentive_share=rng.uniform(0.4, 0.9),
                   people_within=None if rng.random() < spread_share else rng.uniform(3.0, 10.0), speed_gain=rng.uniform(0.85, 1.1),
                   turn_gain=rng.uniform(0.85, 1.1), delay=rng.random() < 0.3,
                   lidar_noise_m=rng.uniform(0.0, 0.03), lidar_dropout=rng.uniform(0.0, 0.02),
                   elderly_share=rng.uniform(0.0, 0.4), room_exits=rng.choice([0, 0, 1, 2]),
                   events=tuple(e for e in building_events if rng.random() < 0.3))


@dataclass
class Chair:
    """The chair, with the attributes the 2D code reads from SocNavGym's robot."""
    x: float = 0.0
    y: float = 0.0
    orientation: float = 0.0
    vel_x: float = 0.0        # forward speed (m/s)
    vel_a: float = 0.0        # turn rate (rad/s), + is left
    goal_x: float = 0.0
    goal_y: float = 0.0
    radius: float = CHAIR_RADIUS_M


class WebotsScene:
    """A Webots world that looks like the 2D simulator's environment to the rest of the code.
    reset() and step() work like a Gymnasium environment's (step takes the same 3-number
    command and returns the same `info` keys), and `unwrapped` is the scene itself."""

    TIMESTEP = STEP_S
    ROBOT_RADIUS = CHAIR_RADIUS_M
    HUMAN_DIAMETER = PERSON_DIAMETER_M
    MAX_ADVANCE_ROBOT = MAX_SPEED
    MAX_ROTATION = MAX_TURN
    MAX_ADVANCE_HUMAN = behaviours.settings()["speed_max"]   # the fastest anyone walks

    def __init__(self, world: str | Path, walls: str = "lidar", animate: bool = True, camera: bool = False,
                 trail: bool = False, variety: bool = False, crowd: str = "busy", spread_share: float = 0.5,
                 see_through_walls: bool = False, stream: bool = False, crowd_scale: float = 1.0,
                 walkers: str = "rules"):
        """crowd: "busy" puts people near the chair's route (the 2D simulator's way, a stress test),
        "spread" anywhere in the building (more like a real day).
        stream: keep people around the chair and its route, all trip long: new ones appear out of its
        sight, those far away disappear, and when the goal changes (set_goal) the new route fills up.
        crowd_scale: this many times the world's usual number of walking people (at the start, and
        kept up by stream). Off by default, so training and testing stay comparable.
        walkers: "rules" (the 2D simulator's social force) or "learned" (Trajectron++ trained on real
        pedestrian recordings, webots/learned_walk.py) for how ordinary people step."""
        if crowd not in CROWDS:
            raise ValueError(f"crowd must be one of {', '.join(CROWDS)}")
        settings_file = Path(world).with_suffix(".json")
        self.settings = json.loads(settings_file.read_text())
        self.plan = load_floorplan(str((ROOT / self.settings["map"]).resolve()))
        self.walls = walls
        self.animate = animate
        self.variety = variety
        self.crowd_style = crowd
        self.stream_people, self.crowd_scale = stream, crowd_scale
        self.building = re.sub(r"_(v\d+|test)$", "", Path(world).stem)   # fp_hospital_v2 -> fp_hospital
        self.spread_share = spread_share     # with variety: share of trips with spread-out crowds
        self.trip = Variety()

        self.sup = Supervisor()
        self.step_ms = int(round(STEP_S * 1000))
        basic = self.sup.getBasicTimeStep()
        if self.step_ms % int(basic):
            raise ValueError(f"The world's basicTimeStep ({basic} ms) must divide {self.step_ms} ms")
        self.left = self.sup.getDevice("left wheel motor")
        self.right = self.sup.getDevice("right wheel motor")
        for motor in (self.left, self.right):
            motor.setPosition(float("inf"))       # speed control
            motor.setVelocity(0.0)
        self.lidar = self.sup.getDevice("lidar")
        self.lidar.enable(self.step_ms)
        self.camera = self.sup.getDevice("camera")
        if camera:
            self.camera.enable(self.step_ms)
        self.keyboard = self.sup.getKeyboard()
        self.keyboard.enable(self.step_ms)
        self.basic_ms = int(basic)
        self.poll_keys, self._seen_keys = False, set()     # see listen_to_keyboard()
        self.before_moving = None    # a function run just before each step's physics (phones.py aims its cameras there)

        self.chair_node = self.sup.getSelf()
        # Remember the chair as the world file has it (wheels, casters, every joint), so that every
        # trip starts from exactly that state: then a trip ends the same way whatever ran before it.
        self.chair_node.saveState("start")
        self.goal_node = self.sup.getFromDef("GOAL")
        self.ped_nodes = [self.sup.getFromDef(f"PED{i}") for i in range(self.settings["people_slots"])]
        self.ped_nodes = [n for n in self.ped_nodes if n is not None]
        self.obj_nodes = []                  # (pose, box, look) of each event-object slot (make_world.py)
        for i in range(8):
            pose = self.sup.getFromDef(f"OBJ{i}")
            if pose is None:
                break
            self.obj_nodes.append((pose, self.sup.getFromDef(f"OBJ{i}_BOX"), self.sup.getFromDef(f"OBJ{i}_LOOK")))
        self.ped_joints = []
        self._stillness = [1.0] * len(self.ped_nodes)   # per model: 0 walking pose, 1 standing pose
        self._posed = [None] * len(self.ped_nodes)      # the joint angles last sent to each model
        if animate:
            try:
                self.ped_joints = [[n.getField(j) for j in JOINTS] for n in self.ped_nodes]
                if any(f is None for fields in self.ped_joints for f in fields):
                    self.ped_joints = []
            except Exception:  # an older Pedestrian model without the animation fields
                self.ped_joints = []

        self._cameras = None                 # the chair's camera nodes, for only_on_screen()
        self.rider = self.sup.getFromDef("RIDER")    # the person in the chair (worlds made before it have none)
        if self.rider is not None:
            for name, angle in RIDER_POSE.items():
                field = self.rider.getField(name)
                if field is not None:
                    field.setSFFloat(angle)
            self._hide_rider()
        self._screen_only = []               # what only_on_screen() hid: DEF names, or nodes without one
        self.every_frame = []                 # called with ms before each physics step, like the rider's seat (zoneview.py)
        self.only_on_screen(self.goal_node)
        self.trail_points = self.trail_index = None
        if trail:
            self._make_trail()

        # The assistant's people tracker (assist/tracker.py) sees everyone within 8 m, even behind
        # walls. A real chair's sensors cannot, so here it only gets the people in its line of sight
        # (see_through_walls=True for the 2D behaviour).
        import assist.tracker
        if not hasattr(assist.tracker, "everyone_in_range"):
            assist.tracker.everyone_in_range = assist.tracker.all_people
        everyone = assist.tracker.everyone_in_range
        if see_through_walls:
            assist.tracker.all_people = everyone
        else:
            from webots.people import line_of_sight
            assist.tracker.all_people = lambda u: [
                p for p in everyone(u) if line_of_sight(self.plan, (u.robot.x, u.robot.y), (p.x, p.y))]

        if walls == "lidar":
            # The assistant measures walls with assist.sensors.wall_ranges(plan, x, y, heading).
            # In Webots the lidar does that job instead.
            assist.controller.wall_ranges = self.lidar_ranges
        elif walls != "map":
            raise ValueError("walls must be 'lidar' or 'map'")

        self.robot = Chair()
        self.crowd = Crowd(self)
        if walkers == "learned":
            from webots.learned_walk import LearnedWalkers
            self.crowd.learned = LearnedWalkers()
        elif walkers != "rules":
            raise ValueError("walkers must be 'rules' or 'learned'")
        self.crowd.scale = crowd_scale
        self.goals: dict = {}
        self._walkers: dict = {}
        self.ticks = 0
        self.route: list[tuple[float, float]] = []
        self.people_area = self.plan.spawn_cells
        self.time_limit = float(self.settings.get("time_limit_s", 60))
        self.path_length = 0.0
        self._velocity = self._acceleration = (0.0, 0.0)
        self._jerk_sum, self._steps = 0.0, 0

    # ------------------------------------------------------------ what the 2D code reads
    @property
    def unwrapped(self):
        return self

    @property
    def dynamic_humans(self):
        return self.crowd.walking

    @property
    def static_humans(self):
        return self.crowd.standing

    static_interactions: list = []     # groups are in static_humans
    moving_interactions: list = []
    h_l_interactions: list = []

    # ------------------------------------------------------------ sensors
    def lidar_ranges(self, plan=None, x=None, y=None, heading=None) -> np.ndarray:
        """The 16 wall distances the assistant expects (one every 22.5 degrees, anticlockwise from
        straight ahead, capped at 4 m), taken from the lidar. Unlike the 2D wall sensor, the lidar
        also sees people and anything else in the way."""
        image = np.asarray(self.lidar.getRangeImage(), dtype=np.float32)
        n = len(image)
        # Webots orders a 360 degree lidar's rays from straight behind, turning clockwise
        # (from the left of the robot to its right).
        angles = 2 * math.pi * np.arange(N_RAYS) / N_RAYS
        index = np.round((math.pi - angles) / (2 * math.pi / n)).astype(int) % n
        ranges = image[index]
        ranges[~np.isfinite(ranges)] = MAX_RANGE_M
        if self.trip.lidar_noise_m or self.trip.lidar_dropout:   # a less perfect lidar (variety)
            noise = self._noise_rng.normal(0.0, self.trip.lidar_noise_m, N_RAYS)
            ranges = np.maximum(ranges + noise.astype(np.float32), 0.0)
            ranges[self._noise_rng.random(N_RAYS) < self.trip.lidar_dropout] = MAX_RANGE_M
        return np.minimum(ranges, MAX_RANGE_M)

    def camera_image(self) -> np.ndarray | None:
        """The forward camera's picture as an (height, width, 3) RGB array, if enabled."""
        if self.camera.getSamplingPeriod() <= 0:
            return None
        raw = np.frombuffer(self.camera.getImage(), np.uint8)
        return raw.reshape(self.camera.getHeight(), self.camera.getWidth(), 4)[:, :, 2::-1]

    def listen_to_keyboard(self) -> None:
        """For run.py, where a person presses keys. Webots only reports the keys held down at the
        moment it is read, so a quick tap between two 0.1 s steps would be lost: from now on the
        world runs each 0.1 s step in its basic steps (20 ms) and the keyboard is read after each.
        The chair, the people and the sensors behave exactly as before."""
        self.poll_keys = True
        self.keyboard.enable(self.basic_ms)

    def _read_keys(self) -> set[int]:
        pressed = set()
        key = self.keyboard.getKey()
        while key != -1:
            pressed.add(key)
            key = self.keyboard.getKey()
        return pressed

    def keys(self) -> set[int]:
        """The keys pressed since the last call (with listen_to_keyboard) or held now."""
        pressed = self._seen_keys | self._read_keys()
        self._seen_keys = set()
        return pressed

    def _advance(self) -> None:
        """Run the world for one step (0.1 s)."""
        if not self.poll_keys:
            if self.rider is not None:
                self._seat_rider(self.step_ms)
            for follow in self.every_frame:
                follow(self.step_ms)
            if self.sup.step(self.step_ms) == -1:
                raise SystemExit("Webots was closed")
            return
        for _ in range(self.step_ms // self.basic_ms):
            if self.rider is not None:          # every frame, so the rider never lags behind the chair
                self._seat_rider(self.basic_ms)
            for follow in self.every_frame:     # and whatever is drawn with the chair
                follow(self.basic_ms)
            if self.sup.step(self.basic_ms) == -1:
                raise SystemExit("Webots was closed")
            self._seen_keys |= self._read_keys()

    def hold_still(self) -> None:
        """One step (0.1 s) with nothing moving and the trip clock stopped: run.py's pause."""
        self.left.setVelocity(0.0)
        self.right.setVelocity(0.0)
        if self.before_moving is not None:
            self.before_moving()
        self._advance()
        self.start_time += self.step_ms / 1000

    # ------------------------------------------------------------ episodes
    def reset(self, *, seed: int | None = None, options=None):
        rng = random.Random(seed)
        # Its own random numbers, so that switching variety on does not change the starts and goals.
        events = BUILDING_EVENTS.get(self.building, ())
        self.trip = Variety.draw(random.Random(f"variety {seed}"), self.spread_share, events) \
            if self.variety else Variety()
        if self.crowd_style == "events":
            self.trip = Variety(**{**self.trip.__dict__, **EVENTS, "events": events})
        ev_rng = random.Random(f"events {seed}")
        start, goal, route = self._start_and_goal(rng)
        self.route = [start] + list(route)
        heading = math.atan2(route[0][1] - start[1], route[0][0] - start[0]) if route else 0.0
        heading += rng.uniform(-0.5, 0.5)
        self.robot = Chair(start[0], start[1], heading, goal_x=goal[0], goal_y=goal[1])

        s = self.settings
        within = s.get("people_within") if self.crowd_style == "busy" else None
        if within and self.variety:
            within = self.trip.people_within
        self.people_area = self.plan.cells_near(self.route, within) if within else self.plan.spawn_cells
        if len(self.people_area) == 0:
            self.people_area = self.plan.spawn_cells
        from envs.pedestrians import chair_can_reach_goal
        for _ in range(20):                # nobody standing where the chair cannot get past
            self.goals, self._walkers, self.ticks = {}, {}, 0
            n_walking, n_standing, n_groups = (rng.randint(*s["walking"]), rng.randint(*s["standing"]),
                                               rng.randint(*s["groups"]))
            reserve = self.trip.room_exits + sum(EVENT_PEOPLE.get(e, 0) for e in self.trip.events)
            room = len(self.ped_nodes) - n_standing - 3 * n_groups - reserve
            if self.variety:               # busier or quieter, never more people than the world can draw
                n_walking = round(n_walking * self.trip.crowd_scale)
            usual = n_walking
            n_walking = max(0, min(room, round(n_walking * self.crowd_scale)))
            if self.stream_people:         # no crowds in narrow places: at most so many people per square metre
                n_walking = min(n_walking, density_cap(self.people_area))
            self.crowd.reset(rng, n_walking, n_standing, n_groups, self.people_area, [start, goal],
                             attentive_share=self.trip.attentive_share, elderly_share=self.trip.elderly_share)
            self.crowd.base_walkers = usual
            ordinary = {id(q) for q in self.crowd.everyone}     # placed clear of the start and goal
            ev_rng = random.Random(f"events {seed}")
            if "waiting_area" in self.trip.events:      # standing people: before checking the way is clear
                self.crowd.add_waiting_area(ev_rng, self.route, ev_rng.randint(3, 6), limit=self._free_slots())
            if chair_can_reach_goal(self):
                break
        if self.crowd.learned is not None:      # a new trip: the model forgets everyone's history
            self.crowd.learned.reset(seed or 0)
        if self.trip.room_exits:
            self.crowd.add_room_exits(random.Random(f"exits {seed}"), self.route, self.trip.room_exits,
                                      limit=self._free_slots())
        if "meeting_ends" in self.trip.events:
            self.crowd.add_room_exits(ev_rng, self.route, 1, group=ev_rng.randint(4, 8), limit=self._free_slots())
        if "phone_walker" in self.trip.events:
            self.crowd.make_phone_walkers(ev_rng, ev_rng.randint(1, 2))
        if "box_carriers" in self.trip.events:
            self.crowd.make_carriers(ev_rng, ev_rng.randint(2, 3))
        if "two_way_traffic" in self.trip.events:
            self.crowd.make_streams(ev_rng, self.route, 0.6)
        if "slow_patients" in self.trip.events:
            self.crowd.make_slow_patients(ev_rng, ev_rng.randint(1, 3))
        for event in self.trip.events:
            if event in EVENT_OBJECTS and self._free_objects():
                thing = self.crowd.add_object(ev_rng, EVENT_OBJECTS[event], self.route)
                if thing is not None and event == "emergency_trolley":
                    self._add_staff(ev_rng, thing)
        # Nothing added for an event may start on top of the chair's start or goal.
        added = lambda q: id(q) not in ordinary
        self.crowd.standing = [q for q in self.crowd.standing if not added(q) or
                               min(math.dist((q.x, q.y), start), math.dist((q.x, q.y), goal)) > q.width / 2 + 1.0]
        self.crowd.walking = [q for q in self.crowd.walking if not added(q) or
                              min(math.dist((q.x, q.y), start), math.dist((q.x, q.y), goal)) > q.width / 2 + 1.0]
        self._shape_objects()

        self._last_command = np.zeros(3, dtype=np.float32)
        self._noise_rng = np.random.default_rng([7, seed if seed is not None else 0])
        self.chair_node.loadState("start")     # caster angles, wheel turns and speeds as at the start
        self.hide_from_cameras_again()         # (loading the state shows everything to its cameras again)
        for motor in (self.left, self.right):  # (loading the state also resets the motors' mode)
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.chair_node.getField("translation").setSFVec3f([start[0], start[1], 0.002])
        self.chair_node.getField("rotation").setSFRotation([0, 0, 1, heading])
        self.chair_node.resetPhysics()
        self.goal_node.getField("translation").setSFVec3f([goal[0], goal[1], 0.01])
        self._draw_people(snap=True)
        self.sup.step(self.step_ms)
        self._read_chair()
        self._restart_trail()
        self.start_time = self.sup.getTime()
        self.path_length = 0.0
        self._velocity = self._acceleration = (self.robot.vel_x, self.robot.vel_a)
        self._jerk_sum, self._steps = 0.0, 0
        return None, {}

    def askable(self, x: float, y: float) -> bool:
        """Whether the chair could ask whoever is at (x, y) to make way: a walker (people.py Crowd.askable)."""
        return self.crowd.askable(x, y)

    def make_way(self, x: float, y: float, lane) -> bool | None:
        """The chair asks the walker at (x, y) to let it past, clear of `lane` (its way ahead):
        people.py Crowd.make_way."""
        return self.crowd.make_way(x, y, (self.robot.x, self.robot.y), lane)

    def set_goal(self, x: float, y: float) -> None:
        """Send the chair somewhere new in the middle of a trip: the goal marker moves, the route is
        planned again, and (with stream on) people appear along the new route straight away."""
        self.robot.goal_x, self.robot.goal_y = x, y
        self.goal_node.getField("translation").setSFVec3f([x, y, 0.01])
        here = (self.robot.x, self.robot.y)
        self.route = [here] + list(self.plan.route(here, (x, y)) or [(x, y)])
        if self.stream_people:
            self.crowd.stream(force=True)

    def _people(self) -> list:
        return [p for p in self.crowd.everyone if p.kind == "person"]

    def _objects(self) -> list:
        return [p for p in self.crowd.everyone if p.kind != "person"]

    def _free_slots(self) -> int:
        """People the world can still draw (it has a fixed number of pedestrian models)."""
        return max(0, len(self.ped_nodes) - len(self._people()))

    def _free_objects(self) -> int:
        return max(0, len(self.obj_nodes) - len(self._objects()))

    def _add_staff(self, rng, trolley) -> None:
        """Two staff rushing along with the emergency trolley, just behind it on either side."""
        from webots.people import Person
        for side in (-1, 1):
            if not self._free_slots():
                return
            back = trolley._stream[1]
            dx, dy = back[0] - trolley.x, back[1] - trolley.y
            n = math.hypot(dx, dy) or 1.0
            spot = (trolley.x + dx / n * 1.3 - dy / n * 0.5 * side, trolley.y + dy / n * 1.3 + dx / n * 0.5 * side)
            if self.plan.clearance_at(*spot) < 0.4:
                spot = (trolley.x + dx / n * 1.5, trolley.y + dy / n * 1.5)
            staff = Person(max(p.id for p in self.crowd.everyone) + 1, *spot, avoids_robot=False,
                           _walk_speed=trolley._walk_speed, goal_x=trolley.goal_x, goal_y=trolley.goal_y)
            staff._stream = trolley._stream
            self.crowd.walking.append(staff)
            self.goals[staff.id] = self.goals[trolley.id]

    def _shape_objects(self) -> None:
        """Give each OBJ slot the size and colour of the object it shows this trip."""
        from webots.people import OBJECTS
        for node, thing in zip(self.obj_nodes, self._objects()):
            (length, breadth, height), colour = OBJECTS[thing.kind][1], OBJECTS[thing.kind][2]
            node[1].getField("size").setSFVec3f([length, breadth, height])
            node[2].getField("baseColor").setSFColor([float(c) for c in colour.split()])

    def _start_and_goal(self, rng):
        """A start and goal joined by a route of the scenario's length (as floorplan_env.py)."""
        plan = self.plan
        low, high = self.settings.get("goal_distance", (10, 25))
        best = None
        for _ in range(300):
            start, goal = plan.random_point(rng), plan.random_point(rng)
            if not (low <= math.dist(start, goal) <= high):
                continue
            route = plan.route(start, goal)
            if route is None:
                continue
            excess = max(0.0, route_length(start, route) - high)
            if best is None or excess < best[0]:
                best = (excess, start, goal, route)
            if excess == 0.0:
                break
        if best is None:
            raise RuntimeError(f"Could not find a start and goal {low} to {high} m apart in {plan.name}")
        return best[1], best[2], best[3]

    def step(self, command):
        """Drive with [forward 0..1, sideways (ignored), turn -1..1] for 0.1 s."""
        if self.trip.delay:                    # the chair reacts one step late
            command, self._last_command = self._last_command, command
        forward = float(np.clip(command[0], 0.0, 1.0)) * MAX_SPEED * self.trip.speed_gain  # cannot reverse
        turn = float(np.clip(command[2], -1.0, 1.0)) * MAX_TURN * self.trip.turn_gain
        self.left.setVelocity((forward - turn * WHEEL_GAP_M / 2) / WHEEL_RADIUS_M)
        self.right.setVelocity((forward + turn * WHEEL_GAP_M / 2) / WHEEL_RADIUS_M)

        self.crowd.step(STEP_S)
        self.ticks += 1
        if self.stream_people and self.ticks % round(STREAM_EVERY_S / STEP_S) == 0:
            self.crowd.stream()
        self._draw_people()
        before = (self.robot.x, self.robot.y)
        if self.before_moving is not None:
            self.before_moving()
        self._advance()
        self._read_chair()
        self._extend_trail()
        self.path_length += math.dist(before, (self.robot.x, self.robot.y))
        # Jerk (change of acceleration per second) of the forward speed and turn rate, averaged over
        # the trip, as SocNavGym's JERK_AVG, which eval/metrics.py reports as mean_jerk.
        velocity = (self.robot.vel_x, self.robot.vel_a)
        acceleration = tuple((v - u) / STEP_S for v, u in zip(velocity, self._velocity))
        if self._steps:
            self._jerk_sum += math.dist(acceleration, self._acceleration) / STEP_S
        self._velocity, self._acceleration, self._steps = velocity, acceleration, self._steps + 1
        return None, 0.0, *self._outcome()

    def _outcome(self):
        r = self.robot
        people = self.crowd.everyone
        # Centre distance as if everyone were an adult (as SocNavGym reports it): bigger things
        # (a trolley, a forklift) count as that much closer, smaller ones (a pet) as further.
        nearest = min((math.dist((p.x, p.y), (r.x, r.y)) - (p.width - PERSON_DIAMETER_M) / 2 for p in people),
                      default=math.inf)
        hit_person = nearest < CHAIR_RADIUS_M + PERSON_DIAMETER_M / 2
        clearance = self.plan.clearance_at(r.x, r.y)
        hit_wall = clearance < CHAIR_RADIUS_M
        success = math.dist((r.x, r.y), (r.goal_x, r.goal_y)) < 0.5
        elapsed = self.sup.getTime() - self.start_time
        timeout = elapsed >= self.time_limit
        info = {
            "SUCCESS": success and not (hit_person or hit_wall),
            "COLLISION": hit_person or hit_wall,
            "COLLISION_HUMAN": hit_person,
            "COLLISION_WALL": hit_wall,
            "TIMEOUT": timeout and not (success or hit_person or hit_wall),
            "OUT_OF_MAP": False,
            "MINIMUM_DISTANCE_TO_HUMAN": nearest,
            "DISTANCE_TO_WALL": clearance - CHAIR_RADIUS_M,
            "PATH_LENGTH": self.path_length,
            "JERK_AVG": self._jerk_sum / max(1, self._steps - 1),
            "TIME": elapsed,
        }
        if info["SUCCESS"]:
            info["TIME_TO_REACH_GOAL"] = elapsed
        terminated = success or hit_person or hit_wall
        return terminated, timeout and not terminated, info

    # ------------------------------------------------------------ Webots state
    def _read_chair(self) -> None:
        x, y, _ = self.chair_node.getPosition()
        m = self.chair_node.getOrientation()
        heading = math.atan2(m[3], m[0])
        v = self.chair_node.getVelocity()
        self.robot.x, self.robot.y, self.robot.orientation = x, y, heading
        self.robot.vel_x = v[0] * math.cos(heading) + v[1] * math.sin(heading)
        self.robot.vel_a = v[5]

    def _hide_rider(self) -> None:
        """The rider is not something the chair's own lidar and camera should see (they sit inside it):
        hidden from them, so the assistant, the safety layer and the sign reader see what they saw
        before there was a rider. People walked from phones still see them."""
        for name in ("lidar", "camera"):
            device = self.sup.getDevice(name)
            node = self.sup.getFromDevice(device._tag) if device is not None else None
            if node is not None:
                self.rider.setVisibility(node, False)

    def rider_on_view(self, viewpoint, visible: bool) -> None:
        """Show or hide the rider on the 3D view (hidden in first person, which looks from their eyes)."""
        if self.rider is not None and viewpoint is not None:
            self.rider.setVisibility(viewpoint, visible)

    def _seat_rider(self, ms: int) -> None:
        """Put the rider on the seat where the chair will be after the next `ms` of simulation (its
        pose now, moved on by its speed), so the two move as one on every frame rather than the
        rider catching up each 0.1 s step."""
        p, o, v = self.chair_node.getPosition(), self.chair_node.getOrientation(), self.chair_node.getVelocity()
        dt = ms / 1000
        heading = math.atan2(o[3], o[0]) + v[5] * dt
        x, y = p[0] + v[0] * dt, p[1] + v[1] * dt
        c, s = math.cos(heading), math.sin(heading)
        ax, ay, az = RIDER_SEAT
        self.rider.getField("translation").setSFVec3f([x + ax * c - ay * s, y + ax * s + ay * c, p[2] + az])
        self.rider.getField("rotation").setSFRotation([0, 0, 1, heading])

    def only_on_screen(self, node) -> None:
        """Show `node`, something drawn for the rider (the sensor view, the goal post, a path or a
        line on the floor), on the 3D view only: hidden from every camera on the chair, so the people
        walked from phones see the building as it is, and the chair's own camera reads signs off
        the real scene."""
        if node is None:
            return
        if self._cameras is None:
            names = ["camera"] + [f"walker eye {i}" for i in range(1, 5)]
            devices = [self.sup.getDevice(n) for n in names]
            self._cameras = [c for d in devices if d is not None and (c := self.sup.getFromDevice(d._tag)) is not None]
        key = node.getDef() or node
        if key not in self._screen_only:
            self._screen_only.append(key)
        for camera in self._cameras:
            node.setVisibility(camera, False)

    def hide_from_cameras_again(self) -> None:
        """Hide everything only_on_screen() hid once more: loading the chair's saved state at the start
        of a trip, or switching a camera on, shows it all to the chair's cameras again."""
        for key in self._screen_only:
            node = self.sup.getFromDef(key) if isinstance(key, str) else key
            if node is not None:
                for camera in self._cameras or []:
                    node.setVisibility(camera, False)
        if self.rider is not None:
            self._hide_rider()

    def _make_trail(self) -> None:
        """A line on the floor behind the chair, like the trail in the 2D demo. It is added to
        the world while we run (and reused if it is already there), not saved in the .wbt."""
        if self.sup.getFromDef("TRAIL_POSE") is None:
            self.sup.getRoot().getField("children").importMFNodeFromString(-1, TRAIL_NODE)
        self.only_on_screen(self.sup.getFromDef("TRAIL_POSE"))
        line = self.sup.getFromDef("TRAIL").getField("geometry").getSFNode()
        self.trail_points = line.getField("coord").getSFNode().getField("point")
        self.trail_index = line.getField("coordIndex")

    def _restart_trail(self) -> None:
        """Back to a line of two points, both where the chair now is (a line needs two)."""
        if self.trail_points is None:
            return
        for field in (self.trail_index, self.trail_points):     # indices first, so none dangles
            for _ in range(field.getCount() - 2):
                field.removeMF(-1)
        for i in (0, 1):
            self.trail_points.setMFVec3f(i, [self.robot.x, self.robot.y, TRAIL_HEIGHT_M])

    def _extend_trail(self) -> None:
        """Add the chair's position, once it has moved 5 cm since the last point."""
        if self.trail_points is None:
            return
        x, y = self.robot.x, self.robot.y
        if math.dist(self.trail_points.getMFVec3f(-1)[:2], (x, y)) < 0.05:
            return
        self.trail_points.insertMFVec3f(-1, [x, y, TRAIL_HEIGHT_M])     # the point before its index
        self.trail_index.insertMFInt32(-1, self.trail_points.getCount() - 1)

    def _assign_slots(self) -> dict:
        """Which pedestrian model draws whom: each person keeps theirs for as long as they are in the
        scene, so that people appearing and disappearing never change who a model shows. Returns
        {model index: person}."""
        people = self._people()
        n = len(self.ped_nodes)
        taken = {p.slot for p in people if p.slot is not None and p.slot < n}
        by_slot = {p.slot: p for p in people if p.slot in taken}
        free = iter(i for i in range(n) if i not in taken)
        for p in people:
            if p.slot in taken:
                continue
            i = next(free, None)
            p.slot = i
            if i is not None:
                by_slot[i] = p
                self._stillness[i], self._posed[i] = 1.0, None   # a new person: no leftover pose
        return by_slot

    def _draw_people(self, snap: bool = False) -> None:
        from webots.people import OBJECTS
        things = self._objects()
        for i, (pose, _box, _look) in enumerate(self.obj_nodes):
            if i < len(things):
                t = things[i]
                height = OBJECTS[t.kind][1][2]
                pose.getField("translation").setSFVec3f([t.x, t.y, height / 2])
                pose.getField("rotation").setSFRotation([0, 0, 1, t.orientation])
            else:
                pose.getField("translation").setSFVec3f([PARKED[0] + 2 * i, PARKED[1] - 20, 0.5])
        by_slot = self._assign_slots()
        for i, node in enumerate(self.ped_nodes):
            if i in by_slot:
                p = by_slot[i]
                # People standing or talking in a group stand upright, as does a walker who has stopped
                # (blended over STAND_BLEND_S so that nobody snaps between poses). The models are
                # shared out again every trip, so a standing person never keeps a walker's stride.
                still = 1.0 if p.type != "dynamic" or p.speed < STILL_SPEED else 0.0
                w = still if snap else float(np.clip(still, self._stillness[i] - STEP_S / STAND_BLEND_S,
                                                     self._stillness[i] + STEP_S / STAND_BLEND_S))
                self._stillness[i] = w
                phase = (p.gait / GAIT_CYCLE_M * 8) % 8
                k, frac = int(phase), phase - int(phase)
                lift = (GAIT_HEIGHT[k] * (1 - frac) + GAIT_HEIGHT[(k + 1) % 8] * frac) * (1 - w)
                node.getField("translation").setSFVec3f([p.x, p.y, PEDESTRIAN_HEIGHT + lift])
                node.getField("rotation").setSFRotation([0, 0, 1, p.orientation])
                if self.ped_joints:
                    pose = (GAIT[:, k] * (1 - frac) + GAIT[:, (k + 1) % 8] * frac) * (1 - w) + STAND * w
                    if self._posed[i] is None or not np.allclose(pose, self._posed[i], atol=1e-3):
                        for f, angle in zip(self.ped_joints[i], pose):
                            f.setSFFloat(float(angle))
                        self._posed[i] = pose
            else:
                node.getField("translation").setSFVec3f([PARKED[0] + 2 * i, PARKED[1], PEDESTRIAN_HEIGHT])


__all__ = ["WebotsScene", "Keyboard"]
