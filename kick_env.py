"""
kick_env.py
===========

Train the Microduck to kick the orange ball. Ball placed in front,
reward encourages approaching and hitting it.
"""

import numpy as np
import mujoco
from gymnasium import Env
from gymnasium.spaces import Box
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LEG_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
NOMINAL_POSE = np.array([0,0,-0.458, -0.005, 0.453, 0,0,0.458, 0.005, -0.453], dtype=np.float32)
ACTION_SCALE = np.array([0.25,0.25,0.40,0.50,0.40, 0.25,0.25,0.40,0.50,0.40], dtype=np.float32)
MIRROR = np.array([1,1,1,1,1, -1,-1,-1,-1,-1], dtype=np.float32)


class MicroduckKickEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path=None):
        super().__init__()
        self.xml_path = xml_path or str(ROOT / "assets" / "scene_ball.xml")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * 5
        self.frame_skip = 5
        self.max_episode_steps = 1000

        self.leg_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in LEG_JOINT_NAMES])
        self.leg_qvel_adr = np.array([self.model.joint(n).dofadr[0] for n in LEG_JOINT_NAMES])
        self.leg_act_ids = np.array([self.model.actuator(n).id for n in LEG_JOINT_NAMES])
        self.head_act_ids = np.array([self.model.actuator(n).id for n in HEAD_JOINT_NAMES])
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        self.action_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        obs_dim = 45
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.prev_action = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        self.last_ball_dist = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # place robot
        self.data.qpos[0:3] = [0, 0, 0.12]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # place ball in front, slightly random
        self.data.qpos[self.model.nq-7:self.model.nq-4] = [0.25 + self.np_random.uniform(-0.05,0.05),
                                                         self.np_random.uniform(-0.03,0.03), 0.035]
        self.data.qpos[self.model.nq-4:self.model.nq] = [1,0,0,0]
        self.data.qvel[:] = 0
        self.data.qpos[self.leg_qpos_adr] = NOMINAL_POSE
        self.data.ctrl[self.head_act_ids] = 0
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        self.last_ball_dist = np.linalg.norm(self.data.xpos[self.ball_id][:2] - self.data.xpos[self.trunk_id][:2])
        return self._obs(), {}

    def _obs(self):
        trunk_id = self.trunk_id
        xmat = self.data.xmat[trunk_id].reshape(3, 3)
        joint_pos = self.data.qpos[self.leg_qpos_adr] - NOMINAL_POSE
        joint_vel = self.data.qvel[self.leg_qvel_adr]
        ball_rel = self.data.xpos[self.ball_id][:2] - self.data.xpos[trunk_id][:2]
        ball_vel = self.data.qvel[self.model.nv-6:self.model.nv-4]
        obs = np.concatenate([
            self.data.qvel[0:3]*5,
            self.data.qvel[3:6]*0.5,
            xmat[:,2],
            [self.data.xpos[trunk_id][2]],
            [xmat[2,2]],
            joint_pos,
            joint_vel*0.1,
            self.prev_action,
            ball_rel,
            ball_vel,
        ]).astype(np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        target = NOMINAL_POSE + MIRROR * ACTION_SCALE * action
        self.data.ctrl[self.leg_act_ids] = target
        self.data.ctrl[self.head_act_ids] = 0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        trunk_h = self.data.xpos[self.trunk_id][2]
        xmat = self.data.xmat[self.trunk_id].reshape(3, 3)
        up = xmat[2,2]
        ball_dist = np.linalg.norm(self.data.xpos[self.ball_id][:2] - self.data.xpos[self.trunk_id][:2])
        ball_vel = np.linalg.norm(self.data.qvel[self.model.nv-6:self.model.nv-4])

        reward = 0.5 * (self.last_ball_dist - ball_dist)  # approach ball
        reward += 2.0 * ball_vel  # reward hitting/kicking
        reward += 0.3 * up
        reward -= 0.01 * np.sum(action**2)
        self.last_ball_dist = ball_dist

        fallen = trunk_h < 0.04 or up < 0.3
        terminated = fallen
        truncated = self.step_count >= self.max_episode_steps
        self.prev_action = action.copy()
        return self._obs(), float(reward), terminated, truncated, {"ball_dist": ball_dist}
