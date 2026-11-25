"""Train the assistant's policy: copy a cautious rule first, then improve on it with PPO.

    python training/train_assist.py                                  # about 45 minutes on 2 cores
    python training/train_assist.py --demo-steps 5000 --timesteps 20000   # a quick try

Two stages:
  1. Head start (behaviour cloning). The hand-written rule in training/teacher.py
     drives for --demo-steps steps. The policy network is trained to copy its
     actions, and the value network to predict the rewards that followed. Then,
     for --dagger-rounds rounds, the copy drives while the rule labels what it
     should have done, and the network trains on all of it (DAgger).
  2. Reinforcement learning (PPO) for --timesteps decisions (100,000 by default).
     The policy tries small variations on what it does and keeps those that earn
     more reward (training/assist_env.py): reaching the goal sooner, smoothly,
     without collisions.

Each step is one decision: the policy is only asked when someone is on the
route ahead or predicted to come too close, and in between the chair follows
its route on its own. Trips are on all eight scenarios (assist_env.py).

The policy learns how much to slow down and how far to steer off the route, from the
people tracker's predictions and the wall sensor (see assist/controller.py).
Progress is printed every 10,000 steps; the result is saved to
models/assist_ppo.zip, which scripts/demo.py and eval/run_episodes.py use as the
"assist" driver.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from training.assist_env import AssistEnv  # noqa: E402

MODEL_PATH = ROOT / "models" / "assist_ppo.zip"
GAMMA = 0.99
MONITOR_KEYS = ("SUCCESS", "COLLISION", "COLLISION_HUMAN")


class Progress(BaseCallback):
    """Print success and collision rates over the last 100 episodes."""

    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % max(1, self.every // self.training_env.num_envs) == 0 and self.model.ep_info_buffer:
            print(f"{self.num_timesteps:>8} steps  {summary(list(self.model.ep_info_buffer))}  "
                  f"({self.num_timesteps / (time.time() - self.start):.0f} steps/s)", flush=True)
        return True


def summary(episodes: list[dict]) -> str:
    if not episodes:
        return "no trips finished yet"
    success = np.mean([e.get("SUCCESS", 0) for e in episodes])
    person = np.mean([e.get("COLLISION_HUMAN", 0) for e in episodes])
    wall = np.mean([e.get("COLLISION", 0) and not e.get("COLLISION_HUMAN", 0) for e in episodes])
    ret = np.mean([e["r"] for e in episodes])
    return f"return {ret:6.2f}  success {success:5.0%}  hit person {person:5.0%}  hit wall {wall:5.0%}"


def make_envs(n: int, seed: int, teacher=False):
    return make_vec_env(AssistEnv, n_envs=n, seed=seed, env_kwargs={"teacher": teacher},
                        vec_env_cls=SubprocVecEnv if n > 1 else None,
                        monitor_kwargs={"info_keywords": MONITOR_KEYS})


def rollout(steps: int, n_envs: int, seed: int, model: PPO | None = None):
    """Drive for `steps` steps, with the teacher rule (model=None) or with the policy, recording
    the rule's action in every state visited. Returns observations, the rule's actions, and the
    discounted reward that followed each step (for the value network)."""
    env = make_envs(n_envs, seed, teacher="drive" if model is None else "label")
    obs = env.reset()
    observations, actions, returns = [], [], []
    pending = [[] for _ in range(n_envs)]          # (index, reward) per running episode
    episodes = []
    for _ in range(steps // n_envs):
        if model is None:
            act = np.zeros((n_envs, 2), dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
        next_obs, rewards, dones, infos = env.step(act)
        for i in range(n_envs):
            observations.append(obs[i])
            actions.append(infos[i]["teacher_action"])
            returns.append(0.0)
            pending[i].append((len(returns) - 1, float(rewards[i])))
            if dones[i]:
                total = 0.0
                for index, reward in reversed(pending[i]):
                    total = reward + GAMMA * total
                    returns[index] = total
                pending[i] = []
                episodes.append(infos[i]["episode"])
        obs = next_obs
    env.close()
    unfinished = [index for episode in pending for index, _ in episode]   # no return known yet
    keep = np.setdiff1d(np.arange(len(returns)), unfinished)
    print(f"  {'Rule' if model is None else 'Policy'} drove {len(episodes)} trips: {summary(episodes)}", flush=True)
    return (np.asarray(observations, dtype=np.float32)[keep], np.asarray(actions, dtype=np.float32)[keep],
            np.asarray(returns, dtype=np.float32)[keep])


def clone(model: PPO, observations, actions, returns, epochs: int = 30) -> None:
    """Behaviour cloning: train the policy to output the teacher's actions, and the value
    network to predict the returns that followed."""
    policy = model.policy
    obs = torch.as_tensor(observations)
    target_actions = torch.as_tensor(np.clip(actions, -1.0, 1.0))
    target_values = torch.as_tensor(returns)
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(obs))
        losses = []
        for start in range(0, len(obs), 256):
            batch = order[start:start + 256]
            features = policy.extract_features(obs[batch])
            latent_pi, latent_vf = policy.mlp_extractor(features)
            action_loss = ((policy.action_net(latent_pi) - target_actions[batch]) ** 2).mean()
            value_loss = ((policy.value_net(latent_vf).squeeze(-1) - target_values[batch]) ** 2).mean()
            loss = action_loss + 0.1 * value_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append((action_loss.item(), value_loss.item()))
        if epoch % 10 == 9 or epoch == epochs - 1:
            a, v = np.mean(losses, axis=0)
            print(f"  cloning epoch {epoch + 1}: action error {a:.3f}, value error {v:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=100_000, help="reinforcement learning steps")
    parser.add_argument("--demo-steps", type=int, default=40_000, help="teacher steps to copy first (0 = none)")
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="rounds of the policy driving while the rule labels what it should have done")
    parser.add_argument("--dagger-steps", type=int, default=20_000, help="steps per such round")
    parser.add_argument("--envs", type=int, default=2, help="environments run in parallel (one per CPU core)")
    parser.add_argument("--checkpoint-every", type=int, default=25_000, help="save a snapshot every this many steps")
    parser.add_argument("--out", default=str(MODEL_PATH))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward", choices=["old", "proxemic"], default="old",
                        help="old: closer than 0.5 m; proxemic: assist/proxemics.py's zones")
    args = parser.parse_args()
    os.environ["G19_REWARD"] = args.reward           # read by assist_env.py in the training processes
    import training.assist_env as assist_env
    assist_env.REWARD = args.reward

    env = make_envs(args.envs, args.seed)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, learning_rate=1e-4, gamma=GAMMA,
                gae_lambda=0.95, n_epochs=10, clip_range=0.1, target_kl=0.03,
                policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.2},
                seed=args.seed, device="cpu", verbose=0)
    if args.demo_steps:
        print(f"Stage 1: copying the teacher rule ({args.demo_steps:,} steps)...", flush=True)
        data = rollout(args.demo_steps, args.envs, args.seed + 1000)
        clone(model, *data)
        # DAgger: let the copy drive, label the states it reaches with the rule's action, and
        # train on everything so far. This fixes mistakes that only show up when the copy drives.
        for k in range(args.dagger_rounds):
            print(f"  Correction round {k + 1} of {args.dagger_rounds} ({args.dagger_steps:,} steps)...", flush=True)
            more = rollout(args.dagger_steps, args.envs, args.seed + 2000 + k, model)
            data = tuple(np.concatenate([a, b]) for a, b in zip(data, more))
            clone(model, *data)

    out = Path(args.out)
    (out.parent / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.envs), save_path=str(out.parent / "checkpoints"),
                                     name_prefix=out.stem)
    print(f"Stage 2: reinforcement learning ({args.timesteps:,} steps)...", flush=True)
    model.save(out.parent / "checkpoints" / f"{out.stem}_0_steps")   # the cloned policy, before any RL
    model.learn(total_timesteps=args.timesteps, callback=[Progress(every=10_000), checkpoints])
    model.save(out)
    print(f"Saved {out}")
    env.close()


if __name__ == "__main__":
    main()
