"""
train_mma.py - Train Microduck for MMA/kicking
"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from kick_env import MicroduckKickEnv

ROOT = Path(__file__).resolve().parent

def make_env(seed, rank):
    def _init():
        env = MicroduckKickEnv()
        env.reset(seed=seed+rank)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    (ROOT/"models"/"mma").mkdir(parents=True, exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    env = DummyVecEnv([make_env(0,r) for r in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(1000,0)])
    model_path = ROOT/"models"/"microduck_mma_kick.zip"
    if args.resume and model_path.exists():
        model = PPO.load(str(model_path), env=env, device="auto", tensorboard_log=str(ROOT/"logs"))
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, policy_kwargs=dict(net_arch=[256,256]),
                    tensorboard_log=str(ROOT/"logs"), verbose=1, device="auto")
    ckpt = CheckpointCallback(save_freq=max(50000//args.n_envs,1), save_path=str(ROOT/"models"/"mma"), name_prefix="mma")
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(ROOT/"models"/"mma"/"best"), log_path=str(ROOT/"logs"/"eval_mma"),
                           eval_freq=max(20000//args.n_envs,1), n_eval_episodes=4, deterministic=True)
    model.learn(total_timesteps=args.timesteps, callback=[ckpt,eval_cb], tb_log_name="mma_kick")
    model.save(str(model_path))
    env.close(); eval_env.close()
    print("done")

if __name__=="__main__":
    main()
