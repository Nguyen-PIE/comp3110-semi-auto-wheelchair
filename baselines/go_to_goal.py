"""A naive driver that heads straight for the goal and ignores people.

This is NOT one of the project's baselines (those are manual driving, DWA and
fixed blending). It exists so the simulator has something sensible to show
before those are written, and it is a useful lower bound: anything we build
should collide less than this.
"""

from __future__ import annotations

import math

import numpy as np


class GoToGoal:
    """Turn on the spot to face the goal, then drive straight at it."""

    def __init__(self, turn_gain: float = 1.5, slow_radius: float = 2.0, turn_in_place_above: float = 0.5):
        self.turn_gain = turn_gain
        self.slow_radius = slow_radius
        self.turn_in_place_above = turn_in_place_above  # radians; turn on the spot if facing further away

    def reset(self) -> None:
        pass  # nothing to remember between episodes

    def act(self, obs: dict) -> np.ndarray:
        # obs["robot"] = [goal_x, goal_y, goal_threshold, sin, cos, angle_threshold, radius]
        # with the goal given in the chair's own frame (x forward, y left).
        goal_x, goal_y = float(obs["robot"][0]), float(obs["robot"][1])
        return self.steer(goal_x, goal_y, math.hypot(goal_x, goal_y))

    def steer(self, target_x: float, target_y: float, distance_left: float) -> np.ndarray:
        """Command towards a target in the chair's frame (x forward, y left).

        distance_left is how far the chair still has to go, used to slow down
        near the goal.
        """
        heading_error = math.atan2(target_y, target_x)
        turn = float(np.clip(self.turn_gain * heading_error, -1.0, 1.0))
        if abs(heading_error) > self.turn_in_place_above:
            forward = 0.0
        else:
            forward = min(1.0, distance_left / self.slow_radius) * math.cos(heading_error)
        # SocNavGym actions are [forward, sideways, turn], each in [-1, 1].
        # Sideways is ignored for a differential-drive chair.
        return np.array([forward, 0.0, turn], dtype=np.float32)
