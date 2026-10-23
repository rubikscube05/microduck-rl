"""
train.py
========

Train a PPO (Proximal Policy Optimization) agent to walk the Microduck.

This script is intentionally written as a learning resource.  The comments
explain both the Stable-Baselines3 (SB3) API and the ideas behind each step.

High-level picture
------------------
1. Create several copies of `MicroDuckEnv` in parallel (a vectorized env).
2. Give them to PPO.
3. PPO repeatedly:
     a. collects a rollout of experience (states, actions, rewards),
     b. estimates the advantage of each action (GAE),
     c. updates the neural network with a clipped policy-gradient loss,
     d. repeats until the requested number of environment steps is reached.
4. Save the trained weights and the best evaluation checkpoint.

Run:
    ./.venv/bin/python train.py --timesteps 1000000 --n-envs 4
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from microduck_env import MicroDuckEnv

ROOT = Path(__file__).resolve().parent


def make_env(seed: int, rank: int):
    """Return a factory that creates one monitored environment.

    SB3's VecEnv expects a list of callables (not instantiated envs) so it can
    create the environments inside the worker processes/threads.  Even with
    `DummyVecEnv` (single process) it expects the same interface.

    `Monitor` wraps the environment and records episode reward and length, which
    is what you see as `rollout/ep_rew_mean` and `rollout/ep_len_mean` in
    TensorBoard.
    """

    def _init():
        env = MicroDuckEnv()
        # Seed each environment differently so the rollouts are not identical.
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train a PPO policy for the Microduck walking task."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps (not episodes).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments used to collect experience.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from models/microduck_ppo.zip if it exists.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run that only checks that the pipeline works.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # A smoke run is useful after changing the model or reward: it finishes in
    # a few seconds and catches import/shape/API errors.
    if args.smoke:
        args.timesteps = min(args.timesteps, 4096)
        args.n_envs = 1

    # Make sure the output folders exist.
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "best").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Vectorized environment
    # ------------------------------------------------------------------
    # `DummyVecEnv` runs all environments in the current process.  It is the
    # simplest vector env and is fine here because MuJoCo releases the GIL
    # while stepping.  For very large runs, `SubprocVecEnv` uses processes.
    env = DummyVecEnv(
        [make_env(args.seed, rank) for rank in range(args.n_envs)]
    )

    # A separate environment is used for evaluation.  Evaluation should be
    # deterministic and is not used for training updates.
    eval_env = DummyVecEnv([make_env(args.seed + 10_000, 0)])

    model_path = ROOT / "models" / "microduck_ppo.zip"
    tensorboard_dir = str(ROOT / "logs")

    # ------------------------------------------------------------------
    # 2. Create or load the PPO model
    # ------------------------------------------------------------------
    if args.resume and model_path.exists():
        print(f"[train] Resuming from {model_path}")
        model = PPO.load(
            str(model_path),
            env=env,
            device="auto",
            tensorboard_log=tensorboard_dir,
        )
    else:
        # Hyperparameters (a good starting point for small MuJoCo tasks):
        #
        # learning_rate : size of each gradient step.
        # n_steps       : rollout length per environment before each update.
        # batch_size    : minibatch size for the gradient updates.
        # n_epochs      : how many passes over each rollout.
        # gamma         : discount factor; 0.99 looks ~1 second ahead at 100 Hz.
        # gae_lambda    : bias/variance trade-off for advantage estimation.
        # clip_range    : PPO's trust-region limit on policy change.
        # ent_coef      : entropy bonus; encourages exploration.
        # vf_coef       : weight of the value-function loss.
        # max_grad_norm : gradient clipping for stability.
        # net_arch      : two hidden layers of 256 units.
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tensorboard_dir,
            verbose=1,
            seed=args.seed,
            device="auto",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[train] torch={torch.__version__} device={device} "
        f"envs={args.n_envs} timesteps={args.timesteps}"
    )

    # ------------------------------------------------------------------
    # 3. Callbacks
    # ------------------------------------------------------------------
    # Save a checkpoint regularly.  `save_freq` is counted in calls to the
    # callback, and the callback is called once per vector step, so divide by
    # the number of environments to get roughly 50k environment steps.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix="microduck_ppo",
        verbose=1,
    )

    # Periodically evaluate the deterministic policy and keep the best model.
    # `eval_freq` is also in callback calls (vector steps).
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best"),
        log_path=str(ROOT / "logs" / "eval"),
        eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # ------------------------------------------------------------------
    # 4. Learn
    # ------------------------------------------------------------------
    # `reset_num_timesteps=False` when resuming keeps the TensorBoard x-axis
    # continuous instead of restarting from zero.
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="microduck_ppo",
        progress_bar=False,
        reset_num_timesteps=not args.resume,
    )

    # Save the final policy.
    model.save(str(model_path))
    env.close()
    eval_env.close()

    print(f"[train] Saved final model to {model_path}")
    print("[train] Watch it with: ./.venv/bin/python watch.py")
    print("[train] Compare runs with: ./.venv/bin/tensorboard --logdir logs")


if __name__ == "__main__":
    main()
