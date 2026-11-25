"""How close do the walkers come to real people? Tested on a place the model never saw.

Takes every real person in the ETH/UCY test recordings who was seen for 3.2 s (8 frames) and then for
4.8 s more (12 frames), predicts those 4.8 s, and measures how far the prediction ends up from where
they really walked:

  ADE  average distance from the real path over the 4.8 s (metres)
  FDE  distance from where they really were at the end (metres)

Methods:
  straight on   keep the last velocity (a simple yardstick)
  rules         the simulator's old walkers (people.py's social force): straight at the real
                person's destination at their speed, pushed away from others
  learned       Trajectron++, steered towards the destination as in the simulator
                (webots/learned_walk.py: of its suggestions, the one heading best that way)
  learned, any  Trajectron++, one of its suggestions at random (no destination)
  learned, most likely   Trajectron++, its single most likely path (no destination)
  learned, best Trajectron++, the best of 20 suggestions (the usual score in papers)

rules and learned both know where the person was going (where they really were 4.8 s later), as
people in the simulator know the next corner of their route; the others do not.

    python webots/walk_model/offline_eval.py [--scene zara1] [--model models/walk_trajectron]

With SiT data (--scene sit or sitmix, SiT's test scenes) the robot itself is not scored (it was driven
by a person with a joystick), and the people who were near it get their own lines: how well the model
predicts people reacting to a robot, which is what the simulator needs it for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from webots import learned_walk as LW  # noqa: E402
from webots.people import PERSON_WIDTH_M, PUSH_RANGE_M, PUSH_SPEED  # noqa: E402

OBS, PRED, DT = 8, 12, 0.4


def rules_path(here, velocity, goal, others):
    """people.py's _social, moved in 0.1 s steps; others (PRED x M x 2) move as they really did."""
    speed = max(float(np.linalg.norm(velocity)), 0.3)
    pos, out = here.copy(), []
    for k in range(PRED):
        near = others[k][~np.isnan(others[k]).any(axis=1)]
        for _ in range(4):
            d = goal - pos
            n = np.linalg.norm(d)
            v = d / n * speed if n > 1e-6 else np.zeros(2)
            if len(near):
                away = pos - near
                dist = np.linalg.norm(away, axis=1)
                gap = dist - PERSON_WIDTH_M
                m = (dist > 1e-6) & (gap < PUSH_RANGE_M)
                v = v + (away[m] / dist[m, None] * (PUSH_SPEED * (PUSH_RANGE_M - gap[m]) / PUSH_RANGE_M)[:, None]).sum(0)
            pos = pos + v * DT / 4
        out.append(pos.copy())
    return np.array(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="zara1", help="the held-out ETH/UCY place (default zara1)")
    p.add_argument("--model", type=Path, default=LW.MODEL_DIR)
    p.add_argument("--data", type=Path, default=ROOT / "third_party/Trajectron-plus-plus/experiments/processed")
    p.add_argument("--max", type=int, default=0, help="test at most this many (0: all)")
    p.add_argument("--near-robot", type=float, default=3.0, metavar="M",
                   help="SiT: people this close to the robot are also scored on their own (default 3 m)")
    args = p.parse_args(argv)

    import dill
    import torch
    walkers = LW.LearnedWalkers(args.model, samples=20)
    model = walkers.model
    with open(args.data / f"{args.scene}_test.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    model.set_environment(env)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    errors = {k: [] for k in ("straight on", "rules", "learned", "learned, any", "learned, most likely",
                              "learned, best of 20")}
    near = {k: [] for k in errors}                 # SiT: the people near the robot
    sit = args.scene.startswith("sit")
    from webots.walk_model.sit_to_trajectron import is_robot
    for scene in env.scenes:
        for t in range(OBS - 1, scene.timesteps - PRED):
            preds = model.predict(scene, np.array([t]), PRED, num_samples=20, min_history_timesteps=OBS - 1,
                                  min_future_timesteps=PRED, full_dist=False).get(t, {})
            if not preds:
                continue
            likely = model.predict(scene, np.array([t]), PRED, num_samples=1, min_history_timesteps=OBS - 1,
                                   min_future_timesteps=PRED, z_mode=True, gmm_mode=True, full_dist=False)[t]
            modes = model.predict(scene, np.array([t]), PRED, num_samples=LW.SAMPLES, min_history_timesteps=OBS - 1,
                                  min_future_timesteps=PRED, gmm_mode=LW.GMM_MODE, full_dist=False)[t]
            positions = {n: n.get(np.array([t - OBS + 1, t + PRED]), {'position': ['x', 'y']}) for n in scene.nodes}
            robots = [positions[n][OBS - 1] for n in scene.nodes if sit and is_robot(n.id)]
            for node, samples in preds.items():
                if sit and is_robot(node.id):
                    continue
                samples = samples[0]                       # 20 x PRED x 2
                track = positions[node]
                close = any(np.linalg.norm(track[OBS - 1] - r) < args.near_robot for r in robots
                            if np.all(np.isfinite(r)))
                truth, here = track[OBS:], track[OBS - 1]
                velocity = (track[OBS - 1] - track[OBS - 2]) / DT
                goal = truth[-1]
                others = np.stack([positions[o][OBS:] for o in scene.nodes if o is not node], axis=1) \
                    if len(scene.nodes) > 1 else np.full((PRED, 0, 2), np.nan)
                to_goal = goal - here
                dist = np.linalg.norm(to_goal)
                reach = min(dist, max(np.linalg.norm(velocity), 0.3) * DT * LW.LOOK_AHEAD)
                target = here + (to_goal / dist * reach if dist > 1e-6 else 0)
                m = modes[node][0]
                steered = m[np.argmin(np.linalg.norm(m[:, LW.LOOK_AHEAD - 1] - target, axis=1))]
                guesses = {
                    "straight on": here + velocity * DT * np.arange(1, PRED + 1)[:, None],
                    "rules": rules_path(here, velocity, goal, others),
                    "learned": steered,
                    "learned, any": samples[rng.integers(len(samples))],
                    "learned, most likely": likely[node][0, 0],
                }
                for name, path in guesses.items():
                    e = np.linalg.norm(path - truth, axis=1)
                    errors[name].append((e.mean(), e[-1]))
                    if close:
                        near[name].append((e.mean(), e[-1]))
                e = np.linalg.norm(samples - truth[None], axis=2)
                best = int(np.argmin(e.mean(axis=1)))
                errors["learned, best of 20"].append((e[best].mean(), e[best, -1]))
                if close:
                    near["learned, best of 20"].append((e[best].mean(), e[best, -1]))
            if args.max and len(errors["rules"]) >= args.max:
                break
    n = len(errors["rules"])
    print(f"Held-out place: {args.scene}, {n} real people's 4.8 s paths\n")
    print(f"{'method':22} {'ADE (m)':>8} {'FDE (m)':>8}")
    for name, rows in errors.items():
        ade, fde = np.mean(rows, axis=0)
        print(f"{name:22} {ade:8.2f} {fde:8.2f}")
    if sit and near["rules"]:
        print(f"\nOnly the {len(near['rules'])} paths of people within {args.near_robot:g} m of the robot:")
        for name, rows in near.items():
            ade, fde = np.mean(rows, axis=0)
            print(f"{name:22} {ade:8.2f} {fde:8.2f}")


if __name__ == "__main__":
    main()
