import time
from pathlib import Path
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from kick_env import MicroduckKickEnv

ROOT = Path(__file__).resolve().parent
model = PPO.load(str(ROOT/"models"/"microduck_kick.zip"), device="cpu")
env = MicroduckKickEnv()
obs,_ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.azimuth = 140; viewer.cam.elevation = -18; viewer.cam.distance = 1.2
    while viewer.is_running():
        a,_ = model.predict(obs, deterministic=True)
        obs,r,term,trunc,info = env.step(a)
        viewer.cam.lookat[:] = env.data.xpos[env.trunk_id]
        viewer.sync()
        time.sleep(env.dt)
        if term or trunc: obs,_ = env.reset()
env.close()
