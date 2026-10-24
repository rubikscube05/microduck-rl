import argparse
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from kick_env import MicroduckKickEnv

ROOT = Path(__file__).resolve().parent

def load_font(size):
    for name in ("DejaVuSans.ttf","LiberationSans-Regular.ttf"):
        try: return ImageFont.truetype(name,size)
        except: pass
    return ImageFont.load_default()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT/"models"/"microduck_kick.zip"))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default=str(ROOT/"docs"/"images"))
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    model = PPO.load(args.model, device="cpu")
    env = MicroduckKickEnv()
    obs,_ = env.reset(seed=args.seed)
    renderer = mujoco.Renderer(env.model,height=480,width=640)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.azimuth=140; cam.elevation=-18; cam.distance=1.2
    font = load_font(16)
    frames=[]; done=False; step=0
    while not done and step<800:
        a,_ = model.predict(obs,deterministic=True)
        obs,r,term,trunc,info = env.step(a)
        done=term or trunc; step+=1
        if step%5!=0 and not done: continue
        cam.lookat[:]=env.data.xpos[env.trunk_id]
        renderer.update_scene(env.data,camera=cam)
        frame = renderer.render()
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0,0),(640,32)],fill=(15,20,30))
        draw.text((10,6),f"Kick training | t={step*env.dt:.1f}s | ball_dist={info.get('ball_dist',0):.2f}m",font=font,fill=(235,240,245))
        frames.append(img)
    renderer.close(); env.close()
    poster = frames[len(frames)//3]
    poster.save(out/"microduck_kick.png")
    writer = imageio.get_writer(str(out/"microduck_kick.mp4"),fps=20,codec='libx264',quality=8,macro_block_size=None)
    for im in frames: writer.append_data(np.asarray(im))
    writer.close()
    gif_frames = [im.resize((480,360),Image.LANCZOS).convert('P',palette=Image.ADAPTIVE,colors=128) for im in frames]
    gif_frames[0].save(str(out/"microduck_kick.gif"),save_all=True,append_images=gif_frames[1:],duration=50,loop=0,optimize=True)
    print("saved")

if __name__=="__main__":
    main()
