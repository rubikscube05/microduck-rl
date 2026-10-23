"""
record.py
=========

Render a trained policy to MP4 and GIF for the README.

MuJoCo can render offscreen without opening a window, so this works both on a
desktop and on a headless machine with EGL/OSMesa support.

Run:
    ./.venv/bin/python record.py
    ./.venv/bin/python record.py --seed 5000 --width 640 --height 480
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from microduck_env import MicroDuckEnv

ROOT = Path(__file__).resolve().parent


def load_font(size):
    """Use a real font when available, otherwise fall back to PIL's default."""
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description="Record a Microduck policy.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "microduck_ppo.zip"),
        help="Path to the SB3 .zip policy.",
    )
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "docs" / "images"),
        help="Directory for the generated MP4, GIF, and poster frame.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckEnv()
    obs, _ = env.reset(seed=args.seed)

    # The control loop runs at 100 Hz, so render every `stride` control steps
    # to hit the requested video frame rate.
    stride = max(1, round(100 / args.fps))

    renderer = mujoco.Renderer(
        env.model, height=args.height, width=args.width
    )

    # A free camera that follows the trunk gives a stable 3/4 side view of a
    # robot that is walking away from the origin.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 135.0
    cam.elevation = -12.0
    cam.distance = 0.75
    cam.lookat[:] = env.data.xpos[env.trunk_id]

    font = load_font(max(14, args.height // 26))
    frames = []
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if step % stride != 0 and not done:
            continue

        cam.lookat[:] = env.data.xpos[env.trunk_id]
        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()

        # Burn a small HUD into the frame so the recording is self-explanatory.
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        text = (
            f"PPO policy  |  t={step * env.dt:4.1f}s  |  "
            f"x={info['x_position']:4.2f}m"
        )
        draw.rectangle(
            [(0, 0), (args.width, args.height // 14)],
            fill=(15, 20, 30),
        )
        draw.text(
            (10, args.height // 60),
            text,
            font=font,
            fill=(235, 240, 245),
        )
        frames.append(image)

    renderer.close()
    env.close()

    if not frames:
        raise SystemExit("No frames were rendered.")

    # Poster image for the top of the README.
    poster = frames[min(len(frames) - 1, len(frames) // 3)]
    poster.save(out_dir / "microduck_walking.png")

    # MP4: high quality, reasonable size for a repository.
    mp4_path = out_dir / "microduck_walking.mp4"
    writer = imageio.get_writer(
        mp4_path, fps=args.fps, codec="libx264", quality=8,
        macro_block_size=None,
    )
    for image in frames:
        writer.append_data(np.asarray(image))
    writer.close()

    # GIF: GitHub renders GIFs inline, so this is the version shown in the
    # README.  Downscale and reduce the palette to keep the file small.
    gif_width = 480
    gif_height = round(args.height * gif_width / args.width)
    gif_frames = [
        image.resize((gif_width, gif_height), Image.LANCZOS).convert(
            "P", palette=Image.ADAPTIVE, colors=128
        )
        for image in frames
    ]
    gif_path = out_dir / "microduck_walking.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )

    print(f"Saved {mp4_path} ({mp4_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {out_dir / 'microduck_walking.png'}")
    print(
        f"Episode: distance={info['x_position']:.2f} m, "
        f"fell={terminated}, frames={len(frames)}"
    )


if __name__ == "__main__":
    main()
