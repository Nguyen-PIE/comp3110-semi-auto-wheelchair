"""Laser-style distances from the chair to the nearest wall in several directions.

Uses the distance map of the floor plan (or of the room's plan, which includes
its furniture), stepping along each ray by the distance to the nearest wall
(sphere tracing), so each ray takes only a few lookups. On the real chair or in
Isaac Sim this would come from a lidar.
"""

from __future__ import annotations

import math

import numpy as np

N_RAYS = 16          # one every 22.5 degrees, starting straight ahead
MAX_RANGE_M = 4.0


def wall_ranges(plan, x: float, y: float, heading: float,
                n_rays: int = N_RAYS, max_range: float = MAX_RANGE_M) -> np.ndarray:
    """Distance (m) from (x, y) to the first wall along each ray, capped at max_range."""
    ranges = np.full(n_rays, max_range, dtype=np.float32)
    for i in range(n_rays):
        a = heading + 2 * math.pi * i / n_rays
        c, s = math.cos(a), math.sin(a)
        d = 0.0
        while d < max_range:
            gap = plan.clearance_at(x + c * d, y + s * d)
            if gap < 0.03:
                ranges[i] = d
                break
            d += max(gap, 0.05)
    return ranges
