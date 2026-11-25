"""Create a SocNavGym environment for one of the project's scenarios.

Each scenario is base.yaml plus a small override file in envs/configs/.
Floor-plan scenarios (fp_*) also add floorplan.yaml and use a real building
map; a path to any ROS map YAML (e.g. maps/level2.yaml) works as well.
SocNavGym needs a single config file, so the merged config is written to a
temporary file and passed to gym.make.

Usage:
    from envs.make_env import make_env, SCENARIOS
    env = make_env("s1_corridor")
    env = make_env("fp_hospital")
    env = make_env("maps/level2.yaml")
    obs, info = env.reset(seed=10000)
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from pathlib import Path

import gymnasium as gym
import yaml

from envs import pedestrians, socnavgym_fixes  # noqa: F401  (socnavgym_fixes registers SocNavGym-v2 and patches its bugs)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Randomly generated rooms.
ROOM_SCENARIOS = {
    "s1_corridor": "S1 Corridor",
    "s2_ward": "S2 Ward",
    "s3_concourse_low": "S3 Concourse (5 people)",
    "s3_concourse_mid": "S3 Concourse (10 people)",
    "s3_concourse_high": "S3 Concourse (20 people)",
}
# Real building floor plans (maps downloaded by scripts/get_maps.sh).
FLOORPLAN_SCENARIOS = {
    "fp_hospital": "Hospital floor plan",
    "fp_office": "Office floor plan",
    "fp_warehouse": "Warehouse floor plan",
}
SCENARIOS = {**ROOM_SCENARIOS, **FLOORPLAN_SCENARIOS}

# Seed ranges agreed in the project plan. Never use TEST seeds before Week 10.
TRAIN_SEEDS = range(0, 10_000)
VALIDATION_SEEDS = range(10_000, 10_200)
TEST_SEEDS = range(20_000, 20_200)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text()) or {}


def is_map_path(scenario: str) -> bool:
    """True if the scenario is a path to a floor-plan YAML rather than a scenario name."""
    return scenario.endswith((".yaml", ".yml"))


def scenario_title(scenario: str) -> str:
    return SCENARIOS.get(scenario) or f"Floor plan: {Path(scenario).stem}"


def scenario_config(scenario: str) -> dict:
    """Return the merged config dictionary for a scenario name or a floor-plan YAML path."""
    config = _read("base.yaml")
    if is_map_path(scenario):
        config = _deep_merge(config, _read("floorplan.yaml"))
        return _deep_merge(config, {"project": {"floorplan": scenario}})
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {', '.join(SCENARIOS)}, "
                         "or give the path to a floor-plan .yaml file")
    if scenario in FLOORPLAN_SCENARIOS:
        config = _deep_merge(config, _read("floorplan.yaml"))
    return _deep_merge(config, _read(f"{scenario}.yaml"))


def floorplan_path(project: dict) -> Path:
    path = Path(project["floorplan"]).expanduser()
    if not path.is_absolute():
        path = path.resolve() if path.exists() else ROOT / path
    if not path.exists():
        hint = ("Run  bash scripts/get_maps.sh  to download the hospital, office and warehouse maps."
                if "third_party" in str(path) else "Check the path, or create the map with scripts/import_floorplan.py.")
        raise FileNotFoundError(f"Floor plan not found: {path}\n{hint}")
    return path


class GoodScene(gym.Wrapper):
    """Re-draw the scene until the goal is at least `min_distance` metres away and the chair can
    drive there.

    SocNavGym places the chair and its goal anywhere in the room, so the goal is often right
    next to the chair, and sometimes the chair starts boxed in by furniture or by people who
    stand still all episode. Re-drawing uses seeds derived from the original one, so the same
    seed always gives the same scene.
    """

    def __init__(self, env, min_distance: float):
        super().__init__(env)
        self.min_distance = min_distance

    def reset(self, *, seed=None, options=None):
        for attempt in range(200):
            attempt_seed = None if seed is None else (seed if attempt == 0 else seed * 1000 + attempt)
            obs, info = self.env.reset(seed=attempt_seed, options=options)
            u = self.env.unwrapped
            robot = u.robot
            far_enough = math.hypot(robot.goal_x - robot.x, robot.goal_y - robot.y) >= self.min_distance
            if far_enough and pedestrians.chair_can_reach_goal(u):
                break
        return obs, info


def make_env(scenario: str = "s1_corridor", render: bool = False, overrides: dict | None = None):
    """Create the environment for a scenario name, or for a floor-plan YAML path.

    render=True opens a window when env.render() is called.
    overrides changes config values on top of the scenario, in the same layout as
    the YAML files, for example {"env": {"max_dynamic_humans": 4}}.
    """
    config = scenario_config(scenario)
    if overrides:
        config = _deep_merge(config, overrides)
    project = config.pop("project", {})  # our own settings, not SocNavGym's
    fd, path = tempfile.mkstemp(prefix=f"{Path(scenario).stem}_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config, f)
    render_mode = "human" if render else "rgb_array"

    if project.get("floorplan"):
        from envs import floorplan_env  # noqa: F401  (registers WheelchairFloorPlan-v0)
        env = gym.make(
            "WheelchairFloorPlan-v0",
            config=path,
            render_mode=render_mode,
            floorplan=str(floorplan_path(project)),
            min_goal_distance=project.get("min_goal_distance", 10),
            max_goal_distance=project.get("max_goal_distance", 25),
            people_within=project.get("people_within", 6),
            pixels_per_metre=project.get("pixels_per_metre", 30),
            disable_env_checker=True,
        )
        env.unwrapped._behaviour = project.get("behaviour")   # what the walking people are like
        return GoodScene(env, 0.0)   # the floor-plan env keeps the goal 10 to 25 m away itself

    env = gym.make("SocNavGym-v2", config=path, render_mode=render_mode, disable_env_checker=True)
    env.unwrapped._behaviour = project.get("behaviour")   # what the walking people are like
    return GoodScene(env, project.get("min_goal_distance", 0.0))


def is_floorplan(env) -> bool:
    return hasattr(env.unwrapped, "plan")
