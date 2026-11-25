"""Play recorded pedestrian tracks (from the SiT dataset) back in the simulator.

    python scripts/sit_tracks.py path/to/SiT/labels --out data/sit/tracks.csv
    python webots/run.py fp_hospital --replay data/sit/tracks.csv 6 --tracks

A few of each trip's walkers are taken over by real people: each follows one recorded track,
step for step, at the speed, with the stops, turns and pace changes that person had. The track
is moved to a free spot on the building's floor and turned to fit it (the recording was made
somewhere else), and cut short where it would run into a wall. When a track ends the person
carries on as an ordinary walker. The chair and the other people are not part of the recording,
so a replayed person does not react to them (the simulator still never lets anyone step into the
chair). With `--tracks` the track the chair predicts for each of them is drawn beside the one they
really walk, which is how SiT scores trajectory prediction.

Tracks file: id,t,x,y (metres and seconds), as scripts/sit_tracks.py writes it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_TRACK_S = 4.0          # a track shorter than this is not worth replaying
CLEARANCE_M = 0.4          # a replayed person's centre keeps this far from walls
SPACING_M = 0.5            # and this far from each other at the start
START_CLEAR_M = 3.0         # recorded people start at least this far from the chair
TRIES = 300                # places and turns tried for each track


@dataclass
class Track:
    """One recorded person: positions at times (seconds from the track's start)."""
    id: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def length_s(self) -> float:
        return float(self.t[-1])

    def speed(self) -> float:
        d = np.hypot(np.diff(self.x), np.diff(self.y)).sum()
        return float(d / max(self.t[-1], 1e-6))


def load(path) -> list[Track]:
    """The tracks in a CSV of id,t,x,y that last at least MIN_TRACK_S, each with its own start as time 0."""
    table = pd.read_csv(path)
    tracks = []
    for key, g in table.sort_values("t").groupby("id"):
        t = g["t"].to_numpy(float)
        if len(t) < 5 or t[-1] - t[0] < MIN_TRACK_S:
            continue
        tracks.append(Track(str(key), t - t[0], g["x"].to_numpy(float), g["y"].to_numpy(float)))
    return tracks


class Replayed:
    """A recorded track put into the world: position at time `now` (seconds since the trip began)."""

    def __init__(self, track: Track, x: np.ndarray, y: np.ndarray, start: float):
        self.track, self.x, self.y, self.start = track, x, y, start
        self.held = 0.0            # seconds in a row held up by someone in the way (people.py)

    def position(self, now: float):
        """Where the person is at `now`, or None once the track has ended."""
        u = now - self.start
        if u < 0:
            return float(self.x[0]), float(self.y[0])
        if u > self.track.t[-1]:
            return None
        return float(np.interp(u, self.track.t, self.x)), float(np.interp(u, self.track.t, self.y))

    def velocity(self, person, now: float, dt: float):
        """The velocity that takes `person` to where the recording has them one step on, or None at the end."""
        target = self.position(now + dt)
        if target is None:
            return None
        return np.array([(target[0] - person.x) / dt, (target[1] - person.y) / dt])

    def future(self, now: float, horizon_s: float, step_s: float) -> list[tuple[float, float]]:
        """Where they really walk next: points every `step_s` for `horizon_s`, as far as the track goes."""
        points = []
        for k in range(1, int(round(horizon_s / step_s)) + 1):
            p = self.position(now + k * step_s)
            if p is None:
                break
            points.append(p)
        return points


def fit(plan, track: Track, area, rng: random.Random, taken: list, keep_clear=None):
    """Move `track` onto the plan: turned by a random angle, started at a random spot in `area`
    (route-grid cells), the part that stays clear of walls and of earlier tracks, starting at least
    `keep_clear[1]` m from the point `keep_clear[0]` (the chair). Returns (x, y) arrays, or None."""
    x0, y0 = track.x[0], track.y[0]
    for _ in range(TRIES):
        angle = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(angle), math.sin(angle)
        sx, sy = plan.random_point(rng, area)
        dx, dy = track.x - x0, track.y - y0
        xs, ys = sx + c * dx - s * dy, sy + s * dx + c * dy
        ok = len(xs)
        for i in range(len(xs)):
            if plan.clearance_at(float(xs[i]), float(ys[i])) < CLEARANCE_M:
                ok = i
                break
        if ok < 2 or track.t[ok - 1] < MIN_TRACK_S:
            continue
        if any(math.dist((sx, sy), p) < SPACING_M for p in taken):
            continue
        if keep_clear and math.dist((sx, sy), keep_clear[0]) < keep_clear[1]:
            continue
        return xs[:ok], ys[:ok], ok
    return None


class Replay:
    """Takes some of a trip's walkers over for recorded tracks. Call `begin()` after the trip is set up."""

    def __init__(self, scene, tracks: list[Track], n: int):
        self.scene, self.tracks, self.n = scene, tracks, n
        self.people: list = []

    def begin(self, seed: int) -> int:
        """Choose tracks, fit them to the floor and hand them to walkers. Returns how many were placed."""
        from webots.people import Destination
        scene, plan = self.scene, self.scene.plan
        rng = random.Random(f"replay {seed}")
        walkers = [p for p in scene.crowd.walking
                   if p.kind == "person" and p.player is None and p._traits is not None
                   and p._traits.leader is None and not p._traits.members]
        rng.shuffle(walkers)
        print(f"replay: {len(walkers)} of {len(scene.crowd.walking)} walkers can be replaced, "
              f"{len(self.tracks)} tracks, {self.n} wanted", flush=True)
        order = list(range(len(self.tracks)))
        rng.shuffle(order)
        taken: list = []
        self.people = []
        crowd = scene.crowd
        spare = getattr(scene, "_free_slots", lambda: 0)()    # bodies the world has not used yet
        while len(self.people) < self.n and order:
            placed = None
            for k in list(order)[:4]:                 # a few tracks per person: long ones often do not fit a corridor
                track = self.tracks[k]
                placed = fit(plan, track, scene.people_area, rng, taken, ((scene.robot.x, scene.robot.y), START_CLEAR_M))
                if placed is not None:
                    order.remove(k)
                    break
            if placed is None:
                break
            if walkers:                               # an ordinary walker is taken over ...
                person = walkers.pop()
            elif spare > 0:                           # ... or, in a sparse building, one is added
                spare -= 1
                person = crowd._new_walker(rng, max(p.id for p in crowd.everyone) + 1, (0.0, 0.0),
                                           scene.people_area, 0.0, 0.0)
                person.avoids_robot = False
            else:
                break
            xs, ys, count = placed
            cut = Track(track.id, track.t[:count], track.x[:count], track.y[:count])
            person._replay = Replayed(cut, xs, ys, start=0.0)
            person.x, person.y = float(xs[0]), float(ys[0])
            person.orientation = math.atan2(ys[1] - ys[0], xs[1] - xs[0])
            person._walk_speed = max(0.2, cut.speed())
            person.goal_x, person.goal_y = float(xs[-1]), float(ys[-1])
            scene.goals[person.id] = Destination(person.goal_x, person.goal_y)
            taken.append((float(xs[0]), float(ys[0])))
            self.people.append(person)
        return len(self.people)
