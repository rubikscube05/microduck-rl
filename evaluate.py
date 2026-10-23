"""
evaluate.py
===========

Evaluate a trained policy without opening a viewer.

Evaluation is important in RL because training curves are noisy.  A policy can
look good in TensorBoard but still fail on a fresh set of episodes.  This script
runs deterministic episodes and reports:

    * mean episode reward,
    * mean forward distance,
    * how often the duck fell (terminal state),
    * how often it reached the success distance.

Run:
    ./.venv/bin/python evaluate.py --episodes 20
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from microduck_env import MicroDuckEnv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Microduck policy.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "microduck_ppo.zip"),
        help="Path to the SB3 .zip policy.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--success-distance",
        type=float,
        default=1.0,
        help="Forward distance (metres) counted as a successful episode.",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"No policy found at {args.model}. Train one first.")

    model = PPO.load(args.model, device="cpu")

    # One environment is enough for evaluation.  We use a new seed offset so
    # the test episodes differ from the training seeds.
    env = MicroDuckEnv()

    rewards = []
    distances = []
    falls = 0
    successes = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=5000 + episode)
        done = False
        total_reward = 0.0

        while not done:
            # Deterministic evaluation: always take the most likely action.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        distance = info["x_position"]
        rewards.append(total_reward)
        distances.append(distance)
        falls += int(terminated)
        successes += int(distance >= args.success_distance)

        print(
            f"episode {episode:02d} | reward={total_reward:8.1f} | "
            f"distance={distance:5.2f} m | fell={terminated}"
        )

    env.close()

    print("\nSummary")
    print(f"  mean reward   : {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  mean distance : {np.mean(distances):.2f} m")
    print(f"  falls         : {falls}/{args.episodes}")
    print(
        f"  success rate  : {successes}/{args.episodes} "
        f"(distance >= {args.success_distance} m)"
    )


if __name__ == "__main__":
    main()
