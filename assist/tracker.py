"""Track the people around the chair and predict whether they will come too close.

This plays the role of the chair's perception. Each step it receives the
positions of the people within sensor range, with a little noise added (as a
lidar or camera tracker would give), and keeps the last second of
positions for each person. From those it estimates each person's walking speed
and direction, then predicts the moment of closest approach between that
person and the chair if both keep moving the same way.

The estimates only use positions, never the simulator's true velocities, so the
same code can run on real sensor data later (for example in Isaac Sim).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SENSOR_RANGE_M = 8.0      # people further away than this are not seen
POSITION_NOISE_M = 0.03   # standard deviation of the position noise
HISTORY_STEPS = 10        # positions kept per person (1 s at 10 steps per second)
HORIZON_S = 4.0           # how far ahead collisions are predicted
COLLISION_GAP_M = 0.3     # a predicted gap smaller than this counts as a predicted collision ...
CLEAR_GAP_M = 0.5         # ... and it stays one until the predicted gap is bigger than this, so that
                          # noise in the estimates does not switch it on and off


@dataclass
class Track:
    """What the chair knows about one person."""
    id: int
    x: float               # estimated position (m, map frame)
    y: float
    vx: float              # estimated velocity (m/s, map frame)
    vy: float
    radius: float
    distance: float        # current edge-to-edge gap to the chair (m)
    t_closest: float       # seconds until the closest approach (0 if already moving apart)
    gap_closest: float     # predicted edge-to-edge gap at that moment (m)
    collision_predicted: bool = False   # still getting closer, and predicted to come too close

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def all_people(u) -> list:
    """Every person in the scene: walking, standing alone, and in groups."""
    people = list(u.static_humans) + list(u.dynamic_humans)
    for group in list(u.static_interactions) + list(u.moving_interactions):
        people.extend(group.humans)
    people.extend(i.human for i in u.h_l_interactions)
    return people


class PeopleTracker:
    def __init__(self, noise_m: float = POSITION_NOISE_M):
        self.noise_m = noise_m
        self.history: dict[int, deque] = {}
        self.rng = np.random.default_rng(0)
        self.tracks: list[Track] = []
        self.flagged: set[int] = set()   # people predicted to come too close at the last update

    def reset(self, seed: int | None = None) -> None:
        self.history = {}
        self.tracks = []
        self.flagged = set()
        self.rng = np.random.default_rng(seed)

    def update(self, u) -> list[Track]:
        """Take one sensor reading of the scene and return the tracked people, nearest first."""
        robot = u.robot
        dt = u.TIMESTEP
        chair_r = u.ROBOT_RADIUS
        cvx = robot.vel_x * math.cos(robot.orientation)   # the chair's own velocity
        cvy = robot.vel_x * math.sin(robot.orientation)

        seen = set()
        flagged = set()
        tracks = []
        for person in all_people(u):
            if math.dist((person.x, person.y), (robot.x, robot.y)) > SENSOR_RANGE_M:
                continue
            key = id(person)
            seen.add(key)
            noisy = (person.x + self.rng.normal(0, self.noise_m), person.y + self.rng.normal(0, self.noise_m))
            hist = self.history.setdefault(key, deque(maxlen=HISTORY_STEPS))
            hist.append(noisy)

            # Straight-line fit through the recent positions: its slope is the velocity.
            pts = np.asarray(hist)
            if len(pts) >= 2:
                t = np.arange(len(pts)) * dt
                t_mean = t.mean()
                denom = ((t - t_mean) ** 2).sum()
                vx = float(((t - t_mean) * (pts[:, 0] - pts[:, 0].mean())).sum() / denom)
                vy = float(((t - t_mean) * (pts[:, 1] - pts[:, 1].mean())).sum() / denom)
                x = float(pts[:, 0].mean() + vx * (t[-1] - t_mean))
                y = float(pts[:, 1].mean() + vy * (t[-1] - t_mean))
            else:
                vx = vy = 0.0
                x, y = noisy

            radius = person.width / 2
            # Closest approach, assuming both keep their current velocity.
            px, py = x - robot.x, y - robot.y
            rvx, rvy = vx - cvx, vy - cvy
            rv2 = rvx * rvx + rvy * rvy
            t_star = 0.0 if rv2 < 1e-6 else min(max(-(px * rvx + py * rvy) / rv2, 0.0), HORIZON_S)
            gap_now = math.hypot(px, py) - chair_r - radius
            gap_star = math.hypot(px + rvx * t_star, py + rvy * t_star) - chair_r - radius
            # Someone already close but walking away is not a predicted collision.
            closing = 0.0 < t_star < HORIZON_S
            limit = CLEAR_GAP_M if key in self.flagged else COLLISION_GAP_M
            predicted = closing and gap_star < limit
            if predicted:
                flagged.add(key)
            tracks.append(Track(person.id, x, y, vx, vy, radius, gap_now, t_star, gap_star, predicted))

        for key in list(self.history):   # forget people who left sensor range
            if key not in seen:
                del self.history[key]
        tracks.sort(key=lambda t: t.distance)
        self.tracks = tracks
        self.flagged = flagged
        return tracks
