"""Pick an automatic driver by name. Used by scripts/demo.py and eval/run_episodes.py.

    auto            assist (route_follower if there is no trained model)
    assist          the learned assistant: route following adjusted by the RL policy
    rule            the cautious hand-written rule the assistant first learns from
    go_to_goal      heads straight for the goal (hits walls and furniture)
    route_follower  follows the route around walls and furniture, ignoring people
    random          random commands
"""

from __future__ import annotations

from assist.driver import MODEL_PATH, AssistDriver, RuleDriver
from baselines.go_to_goal import GoToGoal
from baselines.route_follower import RouteFollower

DRIVERS = ["auto", "assist", "rule", "go_to_goal", "route_follower", "random"]


class RandomDriver:
    def __init__(self, env):
        self.env = env

    def reset(self) -> None:
        pass

    def act(self, obs):
        return self.env.action_space.sample()


def make_driver(name: str, env, model_path=None):
    """Return an object with reset() and act(obs) -> action. model_path: the assist driver's policy
    (default models/assist_ppo.zip)."""
    if name == "auto":
        name = "assist" if MODEL_PATH.exists() else "route_follower"
    if name == "assist":
        if not MODEL_PATH.exists():
            raise ValueError(f"No trained assistant at {MODEL_PATH}; run python training/train_assist.py")
        return AssistDriver(env) if model_path is None else AssistDriver(env, model_path)
    if name == "rule":
        return RuleDriver(env)
    if name == "go_to_goal":
        return GoToGoal()
    if name == "route_follower":
        return RouteFollower(env)
    if name == "random":
        return RandomDriver(env)
    raise ValueError(f"Unknown driver '{name}'. Choose from: {', '.join(DRIVERS)}")
