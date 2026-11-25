"""The environment the assistant's policy is trained in (reinforcement learning).

One episode = one trip in one of the scenarios (a floor plan or a generated room,
picked at random) with people walking around. The policy is only asked when
someone is on the route ahead or predicted to come too close; in between, the
chair follows its route on its own and the reward adds up until the next
decision. The policy sees the observation built by assist/controller.py and
outputs two numbers: how much to slow down and how far to steer off the route.
The reward encourages reaching the goal and penalises collisions and getting
close to people:

    +1.0  per metre moved along the route
    +10   reaching the goal
    -30   any collision (person, wall or furniture)
    up to -0.25 per step while driving closer than 0.5 m to a person (scaled by
          the chair's speed, so people walking past a waiting chair cost nothing)
    -0.005 per step (so waiting forever does not pay)
    penalties for changing the adjustment abruptly and for sudden changes in
          the chair's turn rate (smooth driving, no swinging left and right)

With G19_REWARD=proxemic (train_assist.py --reward proxemic) the "closer than 0.5 m" term is
replaced by assist/proxemics.py's zones: an ellipse around the chair's real outline whose front
stretches with speed, up to -0.25 per step deep in the intimate zone and up to -0.02 in the
personal zone, both scaled by the chair's speed, and nothing for people behind the chair.
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from assist import proxemics
from assist.controller import FOLLOW_ROUTE, OBS_SIZE, AssistController
from assist.tracker import all_people
from envs.make_env import TRAIN_SEEDS, make_env
from training.teacher import teacher_action

# Every scenario, floor plans twice as often as the generated rooms.
TRAIN_SCENARIOS = ("fp_hospital", "fp_office", "fp_warehouse") * 2 + (
    "s1_corridor", "s2_ward", "s3_concourse_low", "s3_concourse_mid", "s3_concourse_high")

PROGRESS_PER_M = 1.0
SUCCESS_REWARD = 10.0
COLLISION_REWARD = -30.0   # three times the goal bonus: never trade a collision for speed
CLOSE_GAP_M = 0.5
CLOSE_PENALTY = 0.25
STEP_PENALTY = 0.005
CHANGE_PENALTY = 0.03
TURN_JERK_PENALTY = 0.3   # per (rad/s)^2 of change in the chair's turn rate from one step to the next
REWARD = os.environ.get("G19_REWARD", "old")   # "old" (closer than 0.5 m) or "proxemic" (assist/proxemics.py)


class AssistEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenarios=TRAIN_SCENARIOS, seeds=TRAIN_SEEDS, overrides=None, teacher=False):
        """teacher="drive" (or True) ignores the actions passed to step() and drives with the
        hand-written rule instead (training/teacher.py). teacher="label" drives with the actions
        passed in. Either way the rule's action is reported in info["teacher_action"]."""
        super().__init__()
        self.scenarios = list(scenarios)
        self.seeds = seeds
        self.overrides = overrides
        self.teacher = teacher
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._envs: dict = {}
        self.env = None
        self.controller: AssistController | None = None
        self.prev_xy = (0.0, 0.0)

    def _load(self, name):
        if name not in self._envs:
            env = make_env(name, overrides=self.overrides)
            self._envs[name] = (env, AssistController(env))
        return self._envs[name]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        name = options.get("scenario") or self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        scene_seed = options.get("scene_seed")
        if scene_seed is None:
            scene_seed = int(self.np_random.integers(self.seeds.start, self.seeds.stop))
        self.env, self.controller = self._load(name)
        self.env.reset(seed=int(scene_seed))
        self.controller.reset(int(scene_seed))
        robot = self.env.unwrapped.robot
        self.prev_xy = (robot.x, robot.y)
        return self.controller.observe(), {"scenario": name, "scene_seed": int(scene_seed)}

    def step(self, action):
        """Apply the policy's action, then keep following the route on its own while nobody is on
        the route ahead or predicted to come too close (the policy is not used then, see
        AssistController.needs_adjusting). So each step is a decision that matters, and the
        reward includes everything up to the next one."""
        expert = None
        if self.teacher:
            expert = teacher_action(self.controller, self.env.unwrapped.robot)
            if self.teacher != "label":
                action = expert
        reward, terminated, truncated, info = self._advance(action)
        while not (terminated or truncated) and not self.controller.needs_adjusting():
            more, terminated, truncated, info = self._advance(FOLLOW_ROUTE)
            reward += more
        info = {k: v for k, v in info.items() if k.isupper()}   # drop SocNavGym's bulky extras
        if expert is not None:
            info["teacher_action"] = np.asarray(expert, dtype=np.float32)
        return self.controller.observe_cached(), float(reward), terminated, truncated, info

    def _advance(self, action):
        """One 0.1 s simulator step with `action`, then sense the scene again."""
        u = self.env.unwrapped
        robot = u.robot
        # The route direction before moving, in the map frame.
        fx, fy, _ = self.controller.target_local
        n = math.hypot(fx, fy) or 1.0
        c, s = math.cos(robot.orientation), math.sin(robot.orientation)
        dir_x, dir_y = (c * fx - s * fy) / n, (s * fx + c * fy) / n
        previous = self.controller.last_action.copy()
        previous_turn = float(robot.vel_a)

        command = self.controller.command(action)
        _, _, terminated, truncated, info = self.env.step(command)

        progress = (robot.x - self.prev_xy[0]) * dir_x + (robot.y - self.prev_xy[1]) * dir_y
        self.prev_xy = (robot.x, robot.y)
        reward = PROGRESS_PER_M * progress - STEP_PENALTY
        reward -= CHANGE_PENALTY * float(np.sum((self.controller.last_action - previous) ** 2))
        reward -= TURN_JERK_PENALTY * float(robot.vel_a - previous_turn) ** 2   # swinging left and right
        moving = float(np.clip(robot.vel_x / u.MAX_ADVANCE_ROBOT, 0.0, 1.0))
        if REWARD == "proxemic":
            people = [(p.x, p.y, p.width / 2) for p in all_people(u)]
            gap = proxemics.nearest_gap(people, robot.x, robot.y, robot.orientation, moving)
            reward -= proxemics.penalty(gap, moving)
        else:
            gap = info.get("MINIMUM_DISTANCE_TO_HUMAN", math.inf) - u.ROBOT_RADIUS - u.HUMAN_DIAMETER / 2
            if gap < CLOSE_GAP_M:
                reward -= CLOSE_PENALTY * moving * (CLOSE_GAP_M - max(gap, 0.0)) / CLOSE_GAP_M
        if info.get("SUCCESS"):
            reward += SUCCESS_REWARD
        if info.get("COLLISION") or info.get("OUT_OF_MAP"):
            reward += COLLISION_REWARD
        self.controller.observe()
        return reward, terminated, truncated, info
