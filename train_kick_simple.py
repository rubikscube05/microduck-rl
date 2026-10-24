import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from kick_env import MicroduckKickEnv

ROOT = Path(__file__).resolve().parent

def make_env(seed, rank):
    def _init():
        env = MicroduckKickEnv()
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    (ROOT / "models" / "kick").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    env = DummyVecEnv([make_env(args.seed if hasattr(args,'seed') else 0, r) for r in range(args.n_envs)])
    
    model_path = ROOT / "models" / "microduck_kick.zip"
    if args.resume and model_path.exists():
        model = PPO.load(str(model_path), env=env, tensorboard_log=str(ROOT / "logs"))
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=256,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.015, policy_kwargs=dict(net_arch=[256,256]),
                    tensorboard_log=str(ROOT / "logs"), verbose=1, device="auto")

    cb = CheckpointCallback(save_freq=max(25000//args.n_envs,1), save_path=str(ROOT/"models"/"kick"), name_prefix="kick")
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name="kick")
    except KeyboardInterrupt:
        pass
    model.save(str(model_path))
    env.close()
    print("Saved", model_path)

if __name__ == "__main__":
    import sys
    sys.argv += [] if '--seed' in sys.argv else ['--seed','0']
    # quick parse
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2000000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    # rerun
    (ROOT / "models" / "kick").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    env = DummyVecEnv([make_env(args.seed, r) for r in range(args.n_envs)])
    model_path = ROOT / "models" / "microduck_kick.zip"
    if args.resume and model_path.exists():
        model = PPO.load(str(model_path), env=env, tensorboard_log=str(ROOT / "logs"))
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=256,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.015, policy_kwargs=dict(net_arch=[256,256]),
                    tensorboard_log=str(ROOT / "logs"), verbose=1, device="auto")
    cb = CheckpointCallback(save_freq=max(25000//args.n_envs,1), save_path=str(ROOT/"models"/"kick"), name_prefix="kick")
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name="kick")
    except KeyboardInterrupt:
        pass
    model.save(str(model_path))
    env.close()
    print("Saved", model_path)
