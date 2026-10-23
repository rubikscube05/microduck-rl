"""
train_gui.py
============

Train PPO with a live MuJoCo viewer showing the first environment's
progress in real time. Useful to watch learning happen step by step.

Run:
    ./.venv/bin/python train_gui.py --timesteps 1000000 --n-envs 4
    ./.venv/bin/python train_gui.py --timesteps 500000 --speed 2.0
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from microduck_env import MicroDuckEnv

ROOT = Path(__file__).resolve().parent


class LiveViewerCallback(BaseCallback):
    """Callback that updates a MuJoCo viewer during training."""

    def __init__(self, model_path, update_every=4, render_fps=30, verbose=0):
        super().__init__(verbose)
        self.model_path = model_path
        self.update_every = update_every
        self.render_fps = render_fps
        self.viewer = None
        self.last_render = 0
        self.env = None

    def _on_training_start(self):
        # Create a separate env for visualization
        self.env = MicroDuckEnv(render_mode=None)
        self.env.reset()
        try:
            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -15
            self.viewer.cam.distance = 0.8
            self.viewer.cam.lookat[:] = [0, 0, 0.15]
        except Exception as e:
            print(f"[GUI] Could not open viewer: {e}")
            self.viewer = None

    def _on_step(self):
        if self.viewer is None or not self.viewer.is_running():
            return True

        if self.n_calls % self.update_every == 0:
            now = time.time()
            if now - self.last_render > 1.0 / self.render_fps:
                # Get current observation from env
                try:
                    obs = self.training_env.get_attr('observation')[0]
                except:
                    pass
                # Step visualization env with current policy
                if hasattr(self, '_vis_obs'):
                    action, _ = self.model.predict(self._vis_obs, deterministic=False)
                    self._vis_obs, _, term, trunc, _ = self.env.step(action)
                    if term or trunc:
                        self._vis_obs, _ = self.env.reset()
                else:
                    self._vis_obs, _ = self.env.reset()
                    action, _ = self.model.predict(self._vis_obs, deterministic=False)
                    self._vis_obs, _, term, trunc, _ = self.env.step(action)
                    if term or trunc:
                        self._vis_obs, _ = self.env.reset()

                self.viewer.cam.lookat[0] = self.env.data.xpos[self.env.trunk_id][0]
                self.viewer.cam.lookat[1] = self.env.data.xpos[self.env.trunk_id][1]
                self.viewer.sync()
                self.last_render = now
        return True

    def _on_training_end(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.env is not None:
            self.env.close()


def make_env(seed, rank):
    def _init():
        env = MicroDuckEnv()
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO with live GUI viewer")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="Render FPS")
    return parser.parse_args()


def main():
    args = parse_args()

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    env = DummyVecEnv([make_env(args.seed, rank) for rank in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args.seed + 10000, 0)])

    model_path = ROOT / "models" / "microduck_ppo.zip"

    if args.resume and model_path.exists():
        print(f"[train_gui] Resuming from {model_path}")
        model = PPO.load(str(model_path), env=env, device="auto", tensorboard_log=str(ROOT / "logs"))
    else:
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
            tensorboard_log=str(ROOT / "logs"),
            verbose=1,
            seed=args.seed,
            device="auto",
        )

    print(f"[train_gui] Training with live viewer... Close viewer to stop.")
    viewer_cb = LiveViewerCallback(model_path, update_every=2, render_fps=args.speed)

    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix="microduck_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best"),
        log_path=str(ROOT / "logs" / "eval"),
        eval_freq=max(20000 // args.n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, viewer_cb],
            tb_log_name="microduck_ppo_gui",
            progress_bar=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save(str(model_path))
        env.close()
        eval_env.close()
        print(f"[train_gui] Saved to {model_path}")


if __name__ == "__main__":
    main()
