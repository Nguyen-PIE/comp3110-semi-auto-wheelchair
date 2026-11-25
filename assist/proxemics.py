"""Proxemic zones around the chair (Hall's intimate, personal, social and public distances).

One ellipse per zone, grown out from the chair's real outline (about 1.1 m long and 0.65 m
wide) by the zone's distance. The front of each ellipse stretches with the chair's speed, up to
twice as deep at full speed, because people want more room in front of something coming at
them faster. People are circles (their shoulder width), as everywhere else in the simulator.

zone_gap() is how far a person is from the chair in "zone metres": the zone distance at which
that zone's ellipse just touches the person. 0 means touching the outline, under 0.45 m is the
intimate zone, and so on. The reward (training/assist_env.py, --reward proxemic) and the eval
(eval/comfort.py) both use this file, so they always agree on what "too close" means.

People behind the chair (centre behind its back edge) are left out of the reward: the chair
cannot reverse, and driving on takes it away from them.
"""

from __future__ import annotations

import math

import numpy as np

HALF_LENGTH_M = 0.55     # the chair's outline: 1.1 m long ...
HALF_WIDTH_M = 0.325     # ... and 0.65 m wide (envs/configs/base.yaml)
MAX_STRETCH = 1.0        # at full speed the front of each zone is (1 + this) times as deep

ZONES = ("intimate", "personal", "social", "public")
ZONE_EDGES_M = (0.45, 1.2, 3.6)   # intimate < 0.45 m <= personal < 1.2 m <= social < 3.6 m <= public

INTIMATE_PENALTY = 0.25  # per 0.1 s step at full speed, at the outline (as the old CLOSE_PENALTY)
PERSONAL_PENALTY = 0.02  # per step at full speed, at the inner edge of the personal zone

_BISECT_STEPS = 40
_FAR_M = 50.0


def local(px, py, chair_x: float, chair_y: float, heading: float):
    """Person positions in the chair's frame: x forward, y to the left."""
    dx, dy = np.asarray(px, dtype=float) - chair_x, np.asarray(py, dtype=float) - chair_y
    c, s = math.cos(heading), math.sin(heading)
    return c * dx + s * dy, -s * dx + c * dy


def zone_gap(lx, ly, person_radius, speed_share: float) -> np.ndarray:
    """Zone metres (see the top of this file) for people at chair-frame positions (lx, ly).
    speed_share is the chair's forward speed over its top speed (0 to 1). Works on arrays (all
    four arguments broadcast together, so speed_share can be one value per step)."""
    lx, ly, r, k = np.broadcast_arrays(np.asarray(lx, dtype=float), np.asarray(ly, dtype=float),
                                       np.asarray(person_radius, dtype=float), np.asarray(speed_share, dtype=float))
    stretch = np.where(lx > 0, 1.0 + MAX_STRETCH * np.clip(k, 0.0, 1.0), 1.0)

    def inside(d):
        a = HALF_LENGTH_M + d * stretch + r
        b = HALF_WIDTH_M + d + r
        return (lx / a) ** 2 + (ly / b) ** 2 <= 1.0

    lo = np.zeros(lx.shape)
    hi = np.full(lx.shape, _FAR_M)
    touching = inside(lo)
    for _ in range(_BISECT_STEPS):
        mid = (lo + hi) / 2
        ok = inside(mid)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    gap = np.where(touching, 0.0, hi)
    return np.where(np.isnan(lx) | np.isnan(ly), np.inf, gap)


def zone_of(gap: float) -> str:
    """The zone a zone-metre gap falls in."""
    for name, edge in zip(ZONES, ZONE_EDGES_M):
        if gap < edge:
            return name
    return ZONES[-1]


def zone_index(gaps) -> np.ndarray:
    """0 intimate, 1 personal, 2 social, 3 public, for an array of gaps (inf stays 3)."""
    return np.searchsorted(np.asarray(ZONE_EDGES_M), np.asarray(gaps, dtype=float), side="right")


def behind(lx) -> np.ndarray:
    """True for people whose centre is behind the chair's back edge."""
    return np.asarray(lx, dtype=float) < -HALF_LENGTH_M


def penalty(gap: float, speed_share: float) -> float:
    """The proxemic reward's penalty for one step (a positive number to subtract): strong in the
    intimate zone, gentle in the personal zone, nothing further out, and nothing when stopped."""
    k = float(np.clip(speed_share, 0.0, 1.0))
    if k == 0.0 or not math.isfinite(gap):
        return 0.0
    intimate = INTIMATE_PENALTY * max(0.0, ZONE_EDGES_M[0] - gap) / ZONE_EDGES_M[0]
    personal = PERSONAL_PENALTY * min(1.0, max(0.0, (ZONE_EDGES_M[1] - gap) / (ZONE_EDGES_M[1] - ZONE_EDGES_M[0])))
    return k * (intimate + personal)


def nearest_gap(people, chair_x: float, chair_y: float, heading: float, speed_share: float,
                skip_behind: bool = True) -> float:
    """Zone metres to the nearest person. people: iterable of (x, y, radius)."""
    people = list(people)
    if not people:
        return math.inf
    px, py, pr = (np.array(col, dtype=float) for col in zip(*people))
    lx, ly = local(px, py, chair_x, chair_y, heading)
    gaps = zone_gap(lx, ly, pr, speed_share)
    if skip_behind:
        gaps = np.where(behind(lx), np.inf, gaps)
    return float(gaps.min())
