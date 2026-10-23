"""
view.py
=======

Open the Microduck model in the MuJoCo viewer without any policy.

This is useful for:
  * checking the URDF -> MJCF conversion,
  * inspecting the mesh visuals,
  * experimenting with joint angles and actuators in the viewer's control panel.

By default gravity is turned off so the duck floats in place and you can look at
it.  Pass `--gravity` to let it fall and interact with the floor.

Run:
    ./.venv/bin/python view.py
    ./.venv/bin/python view.py --gravity
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from microduck_env import MicroDuckEnv, NOMINAL_POSE

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="View the Microduck model.")
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Enable gravity (the duck will settle on the floor).",
    )
    args = parser.parse_args()

    # Reuse the same model/data setup as the RL environment so what you see is
    # exactly what the policy sees.
    env = MicroDuckEnv()
    model, data = env.model, env.data

    if not args.gravity:
        model.opt.gravity[:] = 0.0

    # Put the duck in its nominal crouch so the legs are not perfectly straight.
    data.qpos[3] = 1.0  # identity quaternion (w, x, y, z)
    data.qpos[env.leg_qpos_adr] = NOMINAL_POSE
    mujoco.mj_forward(model, data)

    # Let gravity settle the model onto the floor if requested.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        viewer.cam.distance = 0.60
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08]

        while viewer.is_running():
            # Holding the nominal joint targets keeps the pose stable.
            data.ctrl[env.leg_act_ids] = NOMINAL_POSE
            data.ctrl[env.head_act_ids] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

    env.close()


if __name__ == "__main__":
    main()
