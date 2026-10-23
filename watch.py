"""
watch.py
========

Load a trained PPO policy and watch the Microduck walk in the MuJoCo viewer.

This script is the "deployment" side of the RL loop: the environment is the
same one used during training, but instead of learning we simply call
`model.predict(obs)` and send the action to the simulation.

Run:
    ./.venv/bin/python watch.py
    ./.venv/bin/python watch.py --model models/best/best_model.zip
    ./.venv/bin/python watch.py --stochastic
"""

import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from microduck_env import MicroDuckEnv

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Watch the trained Microduck policy.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "microduck_ppo.zip"),
        help="Path to the SB3 .zip policy.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run; 0 means run until the window closes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic mean action.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise SystemExit(
            f"No policy found at {model_path}\n"
            "Train one first: ./.venv/bin/python train.py --timesteps 1000000"
        )

    # The policy only needs a CPU for inference; keeping it on the CPU avoids
    # GPU/CPU copies every control step.
    model = PPO.load(str(model_path), device="cpu")

    # `render_mode=None` because this script opens the viewer itself.  Opening
    # it manually gives us access to the camera controls and a clean shutdown.
    env = MicroDuckEnv(render_mode=None)

    obs, _ = env.reset()

    # `launch_passive` runs the viewer in a separate thread and lets our loop
    # control the simulation.  The context manager closes the window safely.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Frame the tiny robot.  Distances are in metres.
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        viewer.cam.distance = 0.60
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08]

        episode = 0
        total_reward = 0.0

        try:
            while viewer.is_running():
                # `deterministic=True` uses the mean of the policy distribution.
                # `deterministic=False` samples from it, which is useful for
                # seeing how much variety the learned behaviour has.
                action, _ = model.predict(obs, deterministic=not args.stochastic)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # The viewer only redraws when we call sync().
                viewer.sync()

                # Real-time pacing: one env.step simulates `env.dt` seconds.
                time.sleep(env.dt)

                if terminated or truncated:
                    episode += 1
                    print(
                        f"[watch] episode {episode}: "
                        f"distance={info['x_position']:.2f} m, "
                        f"reward={total_reward:.1f}, "
                        f"fell={terminated}"
                    )
                    total_reward = 0.0
                    obs, _ = env.reset()

                    if args.episodes and episode >= args.episodes:
                        break
        except KeyboardInterrupt:
            pass

    env.close()


if __name__ == "__main__":
    main()
