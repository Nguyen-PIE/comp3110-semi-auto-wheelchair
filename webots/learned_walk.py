"""People who walk the way real people do: Trajectron++ trained on real pedestrian data.

Trajectron++ (Salzmann et al., ECCV 2020) is a published model that predicts where people
will walk in the next few seconds, from where they and everyone near them walked over the
last few seconds. It was trained here on ETH/UCY, recordings of about 1,500 real people
walking in 5 public places (webots/walk_model/README.md has the details and the results).

How it moves people in the simulator (`--walkers learned`):

  1. Every 0.4 s (the rate of the recordings) the model looks at the last 3.2 s of everyone
     in the scene, the chair included as if it were a person, and suggests SAMPLES different
     ways each walker could carry on for the next 2.4 s. These are the "realistic" part: how
     fast people walk, how they curve round each other, how they slow down.
  2. The recordings have no floor plans and no destinations, so the model does not know where
     anyone is going. Of its suggestions we keep, for each person, the one that best heads
     towards the next corner of their route without walking into a wall or into someone.
  3. Each 0.1 s step they follow that suggestion. The rules in envs/pedestrians.py still run
     on top of it, as with the old walkers: noticing the chair (attentive or distracted),
     stepping aside, keeping off walls, and speeding up and turning smoothly.

Only ordinary people use the model; pets, children, trolleys and forklifts keep their rules.
The model's random choices come from the trip's seed, so a seed still replays the same way.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJECTRON_DIR = ROOT / "third_party" / "Trajectron-plus-plus" / "trajectron"
MODEL_DIR = ROOT / "models" / "walk_trajectron"

MODEL_DT = 0.4          # s between the recordings' frames; the model thinks in these steps
HISTORY = 8             # frames of history it looks at (3.2 s)
HORIZON = 6             # frames it predicts ahead (2.4 s)
SAMPLES = 20            # suggestions per person each time
GMM_MODE = True         # each suggestion is the most likely path of one of the model's walking "modes"
                        # (go straight, veer left, slow down, ...), without extra jitter on top
LOOK_AHEAD = 3          # frames (1.2 s) ahead used to judge a suggestion
WALL_PENALTY = 3.0      # metres of "off target" a suggestion through a wall or furniture counts as
BUMP_GAP_M = 0.1        # a suggestion that comes closer than this to someone (edge to edge) ...
BUMP_PENALTY = 1.0      # ... counts as this much further off target, per frame
KEEP_PENALTY = 0.2      # per metre a suggestion strays from the one they were following (no dithering)
ATTENTION_M = 3.0       # people pay attention to others within this distance (as in training)

STANDARDIZATION = {'PEDESTRIAN': {
    'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
    'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
    'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}}}


def available() -> str | None:
    """None when the learned walkers can run, otherwise what is missing."""
    if not (TRAJECTRON_DIR / "model" / "trajectron.py").exists():
        return f"Trajectron++ is not in {TRAJECTRON_DIR.parent} (run setup.sh, or webots/walk_model/get_trajectron.sh)"
    if not (MODEL_DIR / "walk_model.pt").exists():
        return f"no trained walking model in {MODEL_DIR} (webots/walk_model/train.sh makes one)"
    try:
        import torch, dill, orjson  # noqa: F401
    except ImportError as e:
        return f"missing Python package {e.name} (pip install dill orjson)"
    return None


class LearnedWalkers:
    """Runs Trajectron++ for a Crowd and hands each walker the velocity it suggests."""

    def __init__(self, model_dir: Path = MODEL_DIR, samples: int = SAMPLES):
        problem = available() if model_dir == MODEL_DIR else None
        if problem:
            raise RuntimeError(f"--walkers learned: {problem}")
        if str(TRAJECTRON_DIR) not in sys.path:
            sys.path.append(str(TRAJECTRON_DIR))
        import torch
        from environment import Environment, Scene
        from model.model_registrar import ModelRegistrar
        from model.trajectron import Trajectron

        torch.set_num_threads(1)    # small batches: one thread is as fast, and repeatable
        self.torch = torch
        self.samples = samples
        with open(model_dir / "config.json") as f:
            self.hyper = json.load(f)
        self.hyper["maximum_history_length"] = HISTORY - 1
        self.env = Environment(node_type_list=['PEDESTRIAN'], standardization=STANDARDIZATION)
        self.env.scenes = [Scene(timesteps=1, dt=MODEL_DT)]   # the model reads its time step from here
        self.env.attention_radius = {(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN): ATTENTION_M}
        registrar = ModelRegistrar(str(model_dir), 'cpu')
        self.model = Trajectron(registrar, self.hyper, None, 'cpu')
        self.model.set_environment(self.env)            # builds the network, then the trained numbers go in
        registrar.model_dict.load_state_dict(torch.load(model_dir / "walk_model.pt", map_location="cpu",
                                                        weights_only=True))
        registrar.eval()
        self.model.set_annealing_params()
        self.reset(0)

    def reset(self, seed: int) -> None:
        """A new trip: forget everyone's history."""
        self.seed = seed
        self.tracks: dict[str, list[tuple[float, float]]] = {}
        self.plans: dict[int, np.ndarray] = {}   # person id -> chosen path, HORIZON points
        self.since = MODEL_DT                    # time since the model last ran
        self.calls = 0

    # ------------------------------------------------------------ each step
    def update(self, crowd, dt: float) -> None:
        """Called once per simulator step, before velocities are worked out."""
        self.since += dt
        if self.since < MODEL_DT - 1e-6:
            return
        self.since = 0.0
        chair = crowd.scene.robot
        present = {f"p{p.id}": (p.x, p.y) for p in crowd.everyone}
        present["chair"] = (chair.x, chair.y)
        for key in list(self.tracks):
            if key not in present:                 # gone (left the scene)
                del self.tracks[key]
        for key, xy in present.items():
            self.tracks.setdefault(key, []).append(xy)
            del self.tracks[key][:-HISTORY]
        walkers = [p for p in crowd.walking if p.kind == "person" and p.goal_x is not None]
        self.plans = self._choose(crowd, walkers, self._predict()) if walkers else {}

    def velocity(self, person) -> np.ndarray | None:
        """The velocity (m/s) the model suggests for `person` now, or None to use the rules."""
        path = self.plans.get(person.id)
        if path is None:
            return None
        k = min(int(self.since / MODEL_DT), HORIZON - 1)
        remaining = (k + 1) * MODEL_DT - self.since
        return (path[k] - np.array([person.x, person.y])) / max(remaining, 0.1)

    # ------------------------------------------------------------ the model
    def _predict(self) -> dict[str, np.ndarray]:
        """SAMPLES suggested paths (SAMPLES x HORIZON x 2) for everyone with 2+ frames of history."""
        import pandas as pd
        from environment import Scene, Node, derivative_of
        columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
        scene = Scene(timesteps=HISTORY, dt=MODEL_DT)
        by_node = {}
        for key, track in self.tracks.items():
            if len(track) < 2:
                continue
            xy = np.array(track)
            vx, vy = derivative_of(xy[:, 0], MODEL_DT), derivative_of(xy[:, 1], MODEL_DT)
            data = pd.DataFrame({('position', 'x'): xy[:, 0], ('position', 'y'): xy[:, 1],
                                 ('velocity', 'x'): vx, ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): derivative_of(vx, MODEL_DT),
                                 ('acceleration', 'y'): derivative_of(vy, MODEL_DT)}, columns=columns)
            node = Node(node_type=self.env.NodeType.PEDESTRIAN, node_id=key, data=data,
                        first_timestep=HISTORY - len(track))
            scene.nodes.append(node)
            by_node[node] = key
        if not scene.nodes:
            return {}
        self.torch.manual_seed(hash((self.seed, self.calls)) & 0x7FFFFFFF)
        self.calls += 1
        out = self.model.predict(scene, np.array([HISTORY - 1]), HORIZON, num_samples=self.samples,
                                 min_history_timesteps=1, min_future_timesteps=0, full_dist=False,
                                 gmm_mode=GMM_MODE)
        return {by_node[n]: p[0] for n, p in out.get(HISTORY - 1, {}).items()}

    def _choose(self, crowd, walkers, predicted) -> dict[int, np.ndarray]:
        """For each walker, the suggestion that best heads for the next corner of their route."""
        plan = crowd.scene.plan
        others = {}   # where everyone else is expected to be: their average suggestion, or standing still
        for p in crowd.everyone:
            paths = predicted.get(f"p{p.id}")
            others[p.id] = (paths.mean(axis=0) if paths is not None
                            else np.tile([p.x, p.y], (HORIZON, 1)), p.width)
        chair = crowd.scene.robot
        chosen = {}
        for person in walkers:
            paths = predicted.get(f"p{person.id}")
            if paths is None:
                continue
            here = np.array([person.x, person.y])
            to_goal = np.array([person.goal_x, person.goal_y]) - here
            dist = float(np.linalg.norm(to_goal))
            reach = min(dist, person._walk_speed * MODEL_DT * LOOK_AHEAD)
            target = here + (to_goal / dist * reach if dist > 1e-6 else 0.0)
            score = np.linalg.norm(paths[:, LOOK_AHEAD - 1] - target, axis=1)
            before = self.plans.get(person.id)
            if before is not None:              # keep to the way they were already going, if it still works
                score += KEEP_PENALTY * np.linalg.norm(paths[:, :LOOK_AHEAD] - before[1:LOOK_AHEAD + 1],
                                                       axis=2).mean(axis=1)
            for k in range(len(paths)):
                for pt in paths[k, :LOOK_AHEAD]:
                    if plan.clearance_at(float(pt[0]), float(pt[1])) < person.width / 2:
                        score[k] += WALL_PENALTY
                        break
            for oid, (path, width) in others.items():
                if oid == person.id or np.linalg.norm(path[0] - here) > ATTENTION_M:
                    continue
                gap = np.linalg.norm(paths[:, :LOOK_AHEAD] - path[None, :LOOK_AHEAD], axis=2) - (person.width + width) / 2
                score += BUMP_PENALTY * (gap < BUMP_GAP_M).sum(axis=1)
            gap = np.linalg.norm(paths[:, :LOOK_AHEAD] - [chair.x, chair.y], axis=2) - (person.width / 2 + chair.radius)
            score += BUMP_PENALTY * (gap < BUMP_GAP_M).sum(axis=1)
            chosen[person.id] = paths[int(np.argmin(score))]
        return chosen
