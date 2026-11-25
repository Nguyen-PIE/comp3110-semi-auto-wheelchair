"""Fixes for SocNavGym v0.2 bugs, applied when this module is imported.

The library itself is not edited, so a fresh `bash setup.sh` stays reproducible.
Imported by envs/make_env.py and envs/floorplan_env.py.

1. Start-up crash: the code reads WALL_SEGMENTSIZE but the attribute is called
   WALL_SEGMENT_SIZE.
2. People walking at about 6 cm/s: the social force model turns forces into a
   velocity by multiplying by the step length. That only gives walking speed
   with SocNavGym's own 1 s steps; with our 0.1 s steps people were ten times
   too slow. The fix rescales to per-second and to the configured speed, so a
   person with nothing in the way walks at max_advance_human.
3. How people walk around the wheelchair. SocNavGym's people froze in front of
   the chair (the pull towards their destination and the push away from the
   chair cancel out), stopped dead whenever they turned (a turn-rate limit meant
   for its own 1 s steps), and swapped destinations after 50 s of real time.
   envs/pedestrians.py replaces this: people walk a route to their destination
   and only get a new one when they arrive, pass the chair on a side chosen
   once (or step aside and wait where there is no room), and change speed and
   direction smoothly. Whether a person is attentive to the chair (70%,
   prob_to_avoid_robot) is drawn once per episode instead of every step.
4. People walking into the chair: a step that would end inside the chair is
   undone, so a collision with a person always involves the chair moving
   towards them, and the chair can always avoid it.

(One more problem, the goal heading check, is fixed in configs/base.yaml.)
"""

from __future__ import annotations

import math

import numpy as np

import socnavgym  # noqa: F401  (registers SocNavGym-v2 with Gymnasium)
import socnavgym.envs.socnavenv_v2 as socnavenv_module
from socnavgym.envs.socnavenv_v2 import SocNavEnv_v2
from socnavgym.envs.utils.human import Human

from envs import pedestrians

if not hasattr(SocNavEnv_v2, "WALL_SEGMENTSIZE"):
    SocNavEnv_v2.WALL_SEGMENTSIZE = property(lambda self: self.WALL_SEGMENT_SIZE)

TURN_RATE = 1e3          # rad/s; people no longer stop to turn (envs/pedestrians.py turns them smoothly)


def _decides_once(self):
    """Whether this person is attentive to the chair, drawn once and then kept for the episode."""
    if getattr(self, "_ignore_chair", False):     # while computing the forces without the chair
        return False
    if not hasattr(self, "_avoids_robot_fixed"):
        self._avoids_robot_fixed = bool(np.random.random() <= self.prob_to_avoid_robot)
    return self._avoids_robot_fixed


if not getattr(SocNavEnv_v2, "_sfm_speed_fixed", False):
    Human.avoids_robot = property(_decides_once)
    Human.has_reached_goal = pedestrians.has_reached_destination
    socnavenv_module.MAX_ORIENTATION_CHANGE = TURN_RATE   # read by SocNavEnv_v2.step on every call
    _original_sfm_velocity = SocNavEnv_v2.compute_sfm_velocity
    _GOAL_WEIGHT = 1 / math.sqrt(3)  # SocNavGym's weight on the "walk to my goal" force

    def _walking_velocity(self, human, *args, **kwargs):
        """The social force model (rescaled to m/s, without the chair), then envs/pedestrians.py."""
        human._ignore_chair = True
        try:
            social = _original_sfm_velocity(self, human, *args, **kwargs) / self.TIMESTEP / _GOAL_WEIGHT
        finally:
            human._ignore_chair = False
        if not hasattr(self, "_walkers"):
            return _limit(social, self.MAX_ADVANCE_HUMAN)
        return pedestrians.velocity(self, human, social)

    def _limit(velocity, top_speed) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        return velocity * (top_speed / speed) if speed > top_speed else velocity

    _original_try_reset = SocNavEnv_v2.try_reset
    _original_step = SocNavEnv_v2.step
    _original_update = SocNavEnv_v2.handle_collision_and_update

    def _try_reset(self, seed=None, options=None):
        result = _original_try_reset(self, seed, options)
        if result[0]:
            pedestrians.prepare(self)
        return result

    def _step(self, action):
        if hasattr(self, "_walkers"):
            pedestrians.before_step(self)
        return _original_step(self, action)

    def _update_without_walking_into_chair(self):
        before = people_positions(self)
        _original_update(self)
        step_back_from_chair(self, before)

    SocNavEnv_v2.compute_sfm_velocity = _walking_velocity
    SocNavEnv_v2.try_reset = _try_reset
    SocNavEnv_v2.step = _step
    SocNavEnv_v2.handle_collision_and_update = _update_without_walking_into_chair
    SocNavEnv_v2._sfm_speed_fixed = True


def _moving_people(env) -> list:
    return env.dynamic_humans + [h for group in env.moving_interactions for h in group.humans]


def people_positions(env) -> dict:
    """Where each moving person is now, to compare with after they move."""
    return {id(h): (h.x, h.y) for h in _moving_people(env)}


def step_back_from_chair(env, before: dict) -> None:
    """Undo this step for anyone it moved into the chair. People already touching the chair
    from where they were are left alone: the chair drove into them, which is a real collision."""
    robot = env.robot
    for human in _moving_people(env):
        if id(human) not in before:
            continue
        if math.hypot(human.x - robot.x, human.y - robot.y) > robot.radius + human.width / 2 + 0.01:
            continue  # too far apart to touch; skips the slower exact check
        if not human.collides(robot):
            continue
        x, y = human.x, human.y
        human.x, human.y = before[id(human)]
        if human.collides(robot):
            human.x, human.y = x, y
