import argparse, time
from pathlib import Path
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from kick_env import MicroduckKickEnv

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT/"models"/"microduck_mma_kick.zip"))
    args = p.parse_args()
    if not Path(args.model).exists():
        print("Train first: ./venv/bin/python train_mma.py")
        return
    model = PPO.load(args.model, device="cpu")
    env = MicroduckKickEnv()
    obs,_ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 120; viewer.cam.elevation = -15; viewer.cam.distance = 1.0
        while viewer.is_running():
            a,_ = model.predict(obs, deterministic=True)
            obs,r,term,trunc,info = env.step(a)
            viewer.cam.lookat[:] = env.data.xpos[env.trunk_id]
            viewer.sync()
            time.sleep(env.dt)
            if term or trunc: obs,_ = env.reset()
    env.close()

if __name__=="__main__":
    main()
