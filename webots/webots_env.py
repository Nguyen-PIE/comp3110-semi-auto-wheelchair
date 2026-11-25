"""The assistant's training environment, with each trip driven in Webots.

It is training/assist_env.py's AssistEnv, unchanged (same observation, action, reward
and "only ask the policy when someone is in the way" logic), except that its
scenarios are Webots worlds. Each WebotsAssistEnv starts its own Webots, on its own
port, the first time it is reset, so several can run side by side (one per CPU core).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEBOTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBOTS_DIR.parent))

from assist.controller import AssistController  # noqa: E402
from envs.make_env import TRAIN_SEEDS  # noqa: E402
from training.assist_env import AssistEnv  # noqa: E402
from webots import launch  # noqa: E402


class WebotsAssistEnv(AssistEnv):
    def __init__(self, world: str = "corridor", port: int = 1234, seeds=TRAIN_SEEDS, walls: str = "lidar",
                 teacher=False, headless: bool | None = None, log_dir: Path | None = None, variety: bool = True,
                 spread_share: float = 0.5, walkers: str = "rules"):
        super().__init__(scenarios=(world,), seeds=seeds, teacher=teacher)
        self.world = world
        self.port = port
        self.walls = walls
        self.headless = sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
            if headless is None else headless
        self.log_dir = log_dir
        self.variety = variety      # random crowds, chair response and lidar noise per trip (bridge.Variety)
        self.spread_share = spread_share
        self.walkers = walkers      # "rules", or "learned": people step as the walking model (learned_walk.py)
        self.process = None

    def _load(self, name):
        if name not in self._envs:
            from webots.run import world_path
            path = world_path(name)
            log = self.log_dir / f"webots_{self.port}.log" if self.log_dir else None
            self.process = launch.start_webots(path, self.port, fast=True, headless=self.headless, log=log)
            launch.connect_to(self.port)
            from webots.bridge import WebotsScene
            scene = WebotsScene(path, walls=self.walls, animate=False, variety=self.variety,
                                spread_share=self.spread_share, walkers=self.walkers)
            self._envs[name] = (scene, AssistController(scene))
        return self._envs[name]

    def close(self):
        launch.stop_webots(self.process)
        self.process = None
