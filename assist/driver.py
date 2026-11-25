"""The assistant as a driver for scripts/demo.py and eval/run_episodes.py.

AssistDriver uses the trained policy (models/assist_ppo.zip). RuleDriver uses the
cautious hand-written rule the policy first learns from (training/teacher.py), as a
baseline with the same sensors and controller but no learning.
"""

from __future__ import annotations

from pathlib import Path

from assist.controller import FOLLOW_ROUTE, AssistController

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "assist_ppo.zip"


class _AdjustedRouteDriver:
    """Route following, adjusted by choose(features) when someone is in the way."""

    def __init__(self, env):
        self.controller = AssistController(env)
        self.adjustment = (1.0, 0.0)   # last (speed factor, steering offset in radians)

    @property
    def route(self):
        return self.controller.route

    @property
    def tracks(self):
        return self.controller.tracks

    def reset(self) -> None:
        self.controller.reset(seed=0)   # fixed sensor noise, so runs are repeatable
        self.adjustment = (1.0, 0.0)

    def choose(self, features):
        raise NotImplementedError

    def act(self, obs):
        features = self.controller.observe()
        # The policy (or rule) is only asked when someone is on the route ahead or predicted to
        # come too close; otherwise the chair just follows its route.
        action = self.choose(features) if self.controller.needs_adjusting() else FOLLOW_ROUTE
        command = self.controller.command(action)
        self.adjustment = (self.controller.speed, self.controller.offset)
        return command


class AssistDriver(_AdjustedRouteDriver):
    """The learned assistant: route following adjusted by the RL policy."""

    def __init__(self, env, model_path: str | Path = MODEL_PATH):
        from stable_baselines3 import PPO  # imported here so the rest of the project does not need it
        super().__init__(env)
        self.model = PPO.load(str(model_path), device="cpu")

    def choose(self, features):
        action, _ = self.model.predict(features, deterministic=True)
        return action


class RuleDriver(_AdjustedRouteDriver):
    """The cautious hand-written rule, with the same sensors and controller as the assistant."""

    def choose(self, features):
        from training.teacher import teacher_action
        return teacher_action(self.controller, self.controller.env.unwrapped.robot)
