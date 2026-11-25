"""Per-episode metrics, matching the definitions in the project plan.

Feed every step's `info` dict to EpisodeTracker.update(), then call summary()
when the episode ends.
"""

from __future__ import annotations

import math

NEAR_MISS_GAP_M = 0.3  # a near miss is passing within 0.3 m of a person (edge to edge)
FREEZE_SECONDS = 2.0   # a freeze is standing still (not moving or turning) for over 2 s
COMPLIANT_M = 0.45     # SocNavGym's personal space rule: the chair's edge at least this far from a person's centre


class EpisodeTracker:
    def __init__(self, env):
        u = env.unwrapped
        self.dt = u.TIMESTEP
        self.robot_radius = u.ROBOT_RADIUS
        self.human_radius = u.HUMAN_DIAMETER / 2
        self.steps = 0
        self.min_gap = math.inf
        self.compliant_steps = 0  # counted here, as Webots does not report PERSONAL_SPACE_COMPLIANCE
        self.min_wall_gap = math.inf  # floor plans only
        self.near_misses = 0
        self._inside_near_miss = False
        self.freezes = 0
        self._stopped_for = 0.0
        self._freeze_counted = False
        self.overrides = 0        # times the user took control from the chair
        self.manual_steps = 0     # steps driven by the user
        self._manual = False
        self.last_info: dict = {}

    def update(self, info: dict, speed: float, turn_rate: float = 0.0, overridden: bool = False) -> None:
        """Call once per step. overridden is True while the user (not the chair) is driving."""
        self.steps += 1
        self.last_info = info

        # SocNavGym reports centre-to-centre distance; convert to the gap between edges.
        centre_dist = info.get("MINIMUM_DISTANCE_TO_HUMAN", math.inf)
        gap = centre_dist - self.robot_radius - self.human_radius
        self.min_gap = min(self.min_gap, gap)
        self.compliant_steps += centre_dist - self.robot_radius >= COMPLIANT_M
        if gap < NEAR_MISS_GAP_M and not self._inside_near_miss:
            self.near_misses += 1
        self._inside_near_miss = gap < NEAR_MISS_GAP_M

        if abs(speed) < 1e-3 and abs(turn_rate) < 1e-3:
            self._stopped_for += self.dt
            if self._stopped_for > FREEZE_SECONDS and not self._freeze_counted:
                self.freezes += 1
                self._freeze_counted = True
        else:
            self._stopped_for = 0.0
            self._freeze_counted = False

        if overridden:
            self.manual_steps += 1
            if not self._manual:  # a new take-over
                self.overrides += 1
        self._manual = overridden

        if "DISTANCE_TO_WALL" in info:  # reported by floor-plan scenarios
            self.min_wall_gap = min(self.min_wall_gap, info["DISTANCE_TO_WALL"])

    def summary(self) -> dict:
        info = self.last_info
        success = bool(info.get("SUCCESS", False))
        if success:
            outcome = "success"
        elif info.get("COLLISION", False) or info.get("OUT_OF_MAP", False):
            outcome = "collision"
        else:
            outcome = "timeout"
        return {
            "outcome": outcome,
            "success": success,
            "collision": outcome == "collision",
            "collision_human": bool(info.get("COLLISION_HUMAN", False)),
            "collision_object": bool(info.get("COLLISION_OBJECT", False)),
            "collision_wall": bool(info.get("COLLISION_WALL", False)),
            "timeout": outcome == "timeout",
            "steps": self.steps,
            "time_s": round(self.steps * self.dt, 2),
            "time_to_goal_s": info.get("TIME_TO_REACH_GOAL"),
            "path_length_m": round(float(info.get("PATH_LENGTH", 0.0)), 2),
            "min_gap_to_person_m": None if math.isinf(self.min_gap) else round(float(self.min_gap), 2),
            "min_gap_to_wall_m": None if math.isinf(self.min_wall_gap) else round(float(self.min_wall_gap), 2),
            "near_misses": self.near_misses,
            "mean_jerk": round(float(info.get("JERK_AVG", 0.0)), 2),
            "freezes": self.freezes,
            "overrides": self.overrides,
            "manual_time_s": round(self.manual_steps * self.dt, 2),
            "personal_space_compliance": round(self.compliant_steps / max(1, self.steps), 3),
        }
