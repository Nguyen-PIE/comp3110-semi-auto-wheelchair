"""What each walking person is like: how fast they walk, whether they are on the phone, stop now
and then, wait a while on arrival, or walk with others.

envs/pedestrians.py decides where people go and how they pass the chair; this module gives each
person their own character on top of that. It is shared by the 2D simulator and Webots, so both
get the same crowd. The numbers are in the `project.behaviour` section of envs/configs/base.yaml.

  - Speed: drawn from a normal distribution per person (a real crowd is a mix of strollers and
    people in a hurry), not the narrow 0.8 to 1.0 m/s it used to be.
  - On the phone: slower, not watching the chair, drifting from side to side, and stopping more
    often to read.
  - Stopping: some people stop for a few seconds now and then (reading a sign, answering a
    message). Everyone may also wait a while on arriving before choosing the next destination.
  - Groups: pairs and threes walk side by side at a slightly easier pace, stop and wait together,
    and hold their places in the group when passing the chair.

`assign()` runs after each reset. During the episode pedestrians.py calls `hold()` (stops and
waits), `sway()` (phone drift) and `follow()` (group members keeping their place).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"
GROUP_SPACING_M = 0.95        # side by side, centre to centre (people 0.72 m across leave a hand's width)
GROUP_PACE = 0.9              # groups walk at this share of the leader's own pace
CATCH_UP = 1.2                # 1/s: how hard a member heads for their place in the group
CATCH_UP_EXTRA_MPS = 0.6      # a member may walk this much faster than the leader to catch up
FAR_FROM_PLACE_M = 2.5        # further than this, a member plans a route to their place
PHONE_SWAY_PERIOD = 1.3       # sway frequency (rad/s) of someone walking while looking at a phone
PLACE_CLEARANCE_M = 0.1       # room around a person standing in their place, beyond their own radius

_settings: dict | None = None


def settings(overrides: dict | None = None) -> dict:
    """The behaviour settings from configs/base.yaml (project.behaviour), plus any overrides."""
    global _settings
    if _settings is None:
        _settings = yaml.safe_load(CONFIG.read_text())["project"]["behaviour"]
    if not overrides:
        return _settings
    return {**_settings, **overrides}


@dataclass
class Traits:
    """One person's character. Stored as `person._traits`."""
    rng: random.Random
    phone: bool = False
    sway: float = 0.0                     # side to side drift as a share of speed (on the phone)
    phase: float = 0.0
    pause_rate: float = 0.0               # chance per second of stopping
    pause_seconds: tuple = (2.0, 8.0)
    pause_left: float = 0.0
    dwell_chance: float = 0.0             # chance of waiting a while on arriving
    dwell_seconds: tuple = (3.0, 15.0)
    dwell_left: float | None = None       # None: not waiting
    leader: object = None                 # the person this one walks with (None for leaders and solo walkers)
    place: tuple = (0.0, 0.0)             # where to be, relative to the leader: (forward, left) in metres
    members: list = field(default_factory=list)   # the leader's group, so the leader waits for stragglers


def give_traits(person, rng: random.Random, cfg: dict | None = None) -> Traits:
    """An ordinary character (waits a while on arriving now and then, nothing else)."""
    cfg = cfg or settings()
    person._traits = Traits(random.Random(rng.random()), dwell_chance=cfg["dwell_chance"],
                            dwell_seconds=tuple(cfg["dwell_seconds"]))
    return person._traits


def in_group(person) -> bool:
    t = traits_of(person)
    return bool(t and (t.leader is not None or t.members))


def draw_speed(rng: random.Random, cfg: dict | None = None) -> float:
    """A preferred walking speed (m/s)."""
    cfg = cfg or settings()
    return float(min(cfg["speed_max"], max(cfg["speed_min"], rng.gauss(cfg["speed_mean"], cfg["speed_sd"]))))


def traits_of(person) -> Traits | None:
    return getattr(person, "_traits", None)


def is_phone(person) -> bool:
    t = traits_of(person)
    return bool(t and t.phone)


def set_attentive(person, attentive: bool) -> None:
    """Whether a person watches for the chair. SocNavGym's people keep this in a private field."""
    if isinstance(getattr(type(person), "avoids_robot", None), property):
        person._avoids_robot_fixed = attentive
    else:
        person.avoids_robot = attentive


def make_phone(person, rng: random.Random, cfg: dict | None = None) -> None:
    """Put a person on the phone: slow, not watching, drifting from side to side, stopping to read."""
    cfg = cfg or settings()
    t = traits_of(person) or give_traits(person, rng, cfg)
    t.phone, t.sway, t.phase = True, rng.uniform(*cfg["phone_sway"]), rng.uniform(0, 2 * math.pi)
    t.pause_rate = rng.uniform(*cfg["phone_pause_rate"])
    t.pause_seconds = tuple(cfg["pause_seconds"])
    person._walk_speed = rng.uniform(*cfg["phone_speed"])
    set_attentive(person, False)


def make_pauser(person, rng: random.Random, rate: tuple, seconds: tuple) -> None:
    t = traits_of(person)
    t.pause_rate, t.pause_seconds = rng.uniform(*rate), tuple(seconds)


# --------------------------------------------------------------------------- setting up
def assign(env, people: list, rng: random.Random, cfg: dict | None = None, plan=None) -> None:
    """Give each of `people` (the ordinary walkers of a new scene) their character. `rng` is a
    random.Random. `plan` is the floor plan (room or building) used to place group members."""
    cfg = cfg or settings()
    for p in people:
        give_traits(p, rng, cfg)
        p._walk_speed = draw_speed(rng, cfg)
        p._last_velocity = None
    for p in people:
        if rng.random() < cfg["phone_share"]:
            make_phone(p, rng, cfg)
    pool = [p for p in people if not is_phone(p)]
    rng.shuffle(pool)
    wanted = round(cfg["group_share"] * len(people))
    grouped = 0
    while pool and wanted - grouped >= 2 and len(pool) >= 2:
        size = 3 if (rng.random() < cfg["group_of_three_share"] and len(pool) >= 3 and wanted - grouped >= 3) else 2
        made = _make_group(env, pool[:size], rng, cfg, plan)
        pool = pool[size:]
        grouped += size if made else 0
    for p in people:        # solo walkers and group leaders may stop; members stop with their leader
        t = traits_of(p)
        if t.leader is None and not t.phone and rng.random() < cfg["pause_share"]:
            make_pauser(p, rng, cfg["pause_rate"], cfg["pause_seconds"])


def _make_group(env, members: list, rng: random.Random, cfg: dict, plan) -> bool:
    """Make members[0] the leader and put the others beside them, if there is room. Returns whether
    the group was made."""
    leader = members[0]
    side = rng.choice([-1, 1])
    slots = [(-0.1, side * GROUP_SPACING_M), (-0.1, -side * GROUP_SPACING_M)]
    if len(members) == 3:   # a three walks abreast, the leader in the middle
        slots = [(-0.1, GROUP_SPACING_M), (-0.1, -GROUP_SPACING_M)]
    placed = []
    for member, slot in zip(members[1:], slots):
        spot = _find_spot(env, member, leader, slot, plan, placed, members)
        if spot is None:
            break
        placed.append((member, slot, spot))
    if not placed:
        return False
    lt = traits_of(leader)
    pace = leader._walk_speed * GROUP_PACE
    leader._walk_speed = pace
    for member, slot, spot in placed:
        mt = traits_of(member)
        mt.leader, mt.place = leader, slot
        member.x, member.y = spot
        member.orientation = leader.orientation
        member._walk_speed = pace
        set_attentive(member, _is_attentive(leader))
        lt.members.append(member)
    return True


def _is_attentive(person) -> bool:
    return bool(person.avoids_robot)


def _slot_point(leader, slot) -> tuple[float, float]:
    c, s = math.cos(leader.orientation), math.sin(leader.orientation)
    return leader.x + c * slot[0] - s * slot[1], leader.y + s * slot[0] + c * slot[1]


def _others(env, skip) -> list:
    people = list(env.dynamic_humans) + list(env.static_humans)
    for group in list(getattr(env, "static_interactions", [])) + list(getattr(env, "moving_interactions", [])):
        people.extend(group.humans)
    return [o for o in people if o is not skip]


def _find_spot(env, member, leader, slot, plan, already, members):
    """A free place for `member` at (or as close as it gets to) their slot beside the leader."""
    radius = member.width / 2
    robot = env.robot
    group = {id(m) for m in members}
    taken = [(a.x, a.y, a.width / 2) for a in _others(env, member) if id(a) not in group]
    taken += [(p[0], p[1], m.width / 2) for m, _s, p in already] + [(leader.x, leader.y, leader.width / 2)]
    for scale in (1.0, 0.8, 0.65):
        for sign in (1, -1):
            x, y = _slot_point(leader, (slot[0], sign * slot[1] * scale))
            if plan.clearance_at(x, y) < radius + PLACE_CLEARANCE_M:
                continue
            if math.dist((x, y), (robot.x, robot.y)) < robot.radius + radius + 0.5:
                continue
            if any(math.dist((x, y), (ox, oy)) < radius + orad + 0.1 for ox, oy, orad in taken):
                continue
            return x, y
    return None


# --------------------------------------------------------------------------- during an episode
def hold(env, person, v: np.ndarray) -> np.ndarray:
    """Zero velocity while the person is stopped (in a pause, or waiting where they arrived);
    otherwise `v` unchanged. Group members stop by following their leader, who stops."""
    t = traits_of(person)
    if t is None:
        return v
    dt = env.TIMESTEP
    if t.dwell_left is not None and t.dwell_left > 0:
        t.dwell_left -= dt
        return np.zeros(2)
    if t.pause_left > 0:
        t.pause_left -= dt
        return np.zeros(2)
    if t.pause_rate and t.leader is None and t.rng.random() < t.pause_rate * dt:
        t.pause_left = t.rng.uniform(*t.pause_seconds)
    return v


def arrived(person, reached: bool) -> bool:
    """Called every step with whether the person is at their destination: whether they are ready to
    choose the next one now (False while they wait there for a while)."""
    t = traits_of(person)
    if t is None:
        return reached
    if not reached:
        if t.dwell_left is not None and t.dwell_left <= 0:   # pushed away while waiting: start again on return
            t.dwell_left = None
        return False
    if t.dwell_left is None:
        t.dwell_left = t.rng.uniform(*t.dwell_seconds) if t.rng.random() < t.dwell_chance else 0.0
    if t.dwell_left > 0:
        return False
    t.dwell_left = None
    return True


def is_stopped(person) -> bool:
    """Whether the person is standing still on purpose (a pause, waiting on arrival, or a group
    leader waiting for the others), so that being still does not count as being stuck."""
    t = traits_of(person)
    return bool(t and (t.pause_left > 0 or (t.dwell_left or 0) > 0 or wait_for_group(person) < 1.0))


def wait_for_group(person) -> float:
    """Share of their pace a group's leader keeps: they slow down, and stop, if the others fall
    behind (more than 2 m away, none of them further than 4 m)."""
    t = traits_of(person)
    if t is None or not t.members:
        return 1.0
    behind = max(math.dist((person.x, person.y), (m.x, m.y)) for m in t.members)
    return min(1.0, max(0.0, 1.0 - (behind - 2.0) / 2.0))


def sway(env, person, v: np.ndarray) -> np.ndarray:
    """People on the phone drift from side to side as they walk."""
    t = traits_of(person)
    if t is None or not t.sway:
        return v
    t.phase += env.TIMESTEP
    return v + np.array([-v[1], v[0]]) * t.sway * math.sin(t.phase * PHONE_SWAY_PERIOD)


def place_of(person) -> tuple[float, float]:
    """Where a group member should be: their slot beside the leader."""
    t = traits_of(person)
    return _slot_point(t.leader, t.place)


def follow(env, person, target) -> np.ndarray:
    """A group member's velocity: the leader's own velocity plus a pull towards their place beside
    the leader. From far away they head for `target`, the next corner of a route to their place."""
    t = traits_of(person)
    leader_v = getattr(t.leader, "_last_velocity", None)
    leader_v = np.zeros(2) if leader_v is None else np.asarray(leader_v)
    here = np.array([person.x, person.y])
    gap = np.asarray(place_of(person)) - here
    distance = float(np.linalg.norm(gap))
    if distance > FAR_FROM_PLACE_M:
        way = np.asarray(target) - here
        length = float(np.linalg.norm(way))
        return way / length * min(person._walk_speed * 1.25, env.MAX_ADVANCE_HUMAN) if length > 1e-6 else way
    v = leader_v + CATCH_UP * gap
    top = min(float(np.linalg.norm(leader_v)) + CATCH_UP_EXTRA_MPS, env.MAX_ADVANCE_HUMAN)
    speed = float(np.linalg.norm(v))
    return v * (top / speed) if speed > top else v
