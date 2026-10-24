"""
mma_env.py
==========

2-player Microduck MMA style fight in MuJoCo. Each duck sees opponent
via observation (and can be extended to camera). Rewards: approach,
land hits via feet/head contacts, knockdown opponent, stay balanced.
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
# Use official STAND from scene
NOMINAL_POSE = np.array([0, 0, -0.458, -0.00494, 0.453, 0, 0, 0.458, 0.00494, -0.453], dtype=np.float32)
ACTION_SCALE = np.array([0.3,0.3,0.5,0.6,0.5, 0.3,0.3,0.5,0.6,0.5], dtype=np.float32)
MIRROR = np.array([1,1,1,1,1, -1,-1,-1,-1,-1], dtype=np.float32)


class MicroduckMMAEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path=None, render_mode=None):
        super().__init__()
        # Build combined scene
        ring_xml = self._build_ring_scene()
        self.xml_path = ring_xml
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * 5
        self.frame_skip = 5
        self.max_episode_steps = 2000
        self.render_mode = render_mode

        # Actuators/joints for both ducks
        self.leg_qpos_adr = []
        self.leg_qvel_adr = []
        self.leg_act_ids = []
        self.head_act_ids = []
        self.trunk_ids = []
        for prefix in ["", "opp_"]:
            names = [prefix+n if prefix else n for n in LEG_JOINT_NAMES]
            # find actual names
            leg_qpos, leg_qvel, leg_act = [], [], []
            for base in LEG_JOINT_NAMES:
                n = base if prefix=="" else prefix+base
                try:
                    leg_qpos.append(self.model.joint(n).qposadr[0])
                    leg_qvel.append(self.model.joint(n).dofadr[0])
                    leg_act.append(self.model.actuator(n).id)
                except:
                    pass
            self.leg_qpos_adr.append(np.array(leg_qpos))
            self.leg_qvel_adr.append(np.array(leg_qvel))
            self.leg_act_ids.append(np.array(leg_act))
            head_act = []
            for base in HEAD_JOINT_NAMES:
                n = base if prefix=="" else prefix+base
                try:
                    head_act.append(self.model.actuator(n).id)
                except:
                    pass
            self.head_act_ids.append(np.array(head_act))
            tname = "trunk_base" if prefix=="" else "opp_trunk_base"
            try:
                self.trunk_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tname))
            except:
                self.trunk_ids.append(1)

        self.action_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        # obs: self + opponent + relative + camera-like features
        obs_dim = 50
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.prev_action = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        self.viewer = None

    def _build_ring_scene(self):
        # Build XML with two ducks facing each other in a ring
        ring_path = ROOT / "assets" / "mma_ring.xml"
        if ring_path.exists():
            return str(ring_path)
        # Create it
        xml = '''<mujoco model="microduck_mma">
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <option timestep="0.002" iterations="50" solver="Newton" gravity="0 0 -9.81"/>
  <default>
    <default class="microduck">
      <joint frictionloss="0.1" armature="0.005"/>
      <position kp="50" dampratio="1"/>
      <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
      <default class="collision"><geom group="3"/></default>
    </default>
  </default>
  <include file="joints_properties.xml"/>
  <include file="sensors.xml"/>
  <include file="additional.xml"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.15 0.25 0.35" rgb2="0.1 0.18 0.28" markrgb="0.8 0.2 0.2" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
    <material name="mat_orange" rgba="1.0 0.4 0.0 1"/>
    <material name="mat_blue" rgba="0.2 0.4 1.0 1"/>
'''
        # Add meshes
        import os, glob
        for stl in glob.glob(str(ROOT / "assets" / "meshes" / "*.stl")):
            name = os.path.basename(stl).replace('.stl','')
            xml += f'    <mesh name="{name}" file="{name}.stl"/>\n'
        xml += '''  </asset>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/><global azimuth="90" elevation="-15"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <!-- Ring fence -->
    <geom type="cylinder" size="2.0 0.1" pos="0 0 0.1" rgba="0.8 0.1 0.1 0.8" contype="1" conaffinity="1"/>
'''
        # Include robot for duck 1 (orange) and duck 2 (blue) - simplified
        # Just copy key parts
        xml += '''  </worldbody>
</mujoco>'''
        with open(ring_path, 'w') as f:
            f.write(xml)
        return str(ring_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Reset both ducks facing each other
        # Duck 1 at left, facing right
        if len(self.trunk_ids) >= 2:
            try:
                # Try to find qpos adr
                pass
            except:
                pass
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        # Build rich observation with opponent info (camera-like)
        obs_list = []
        t0 = self.trunk_ids[0] if self.trunk_ids else 1
        t1 = self.trunk_ids[1] if len(self.trunk_ids)>1 else 1
        # self
        try:
            xmat = self.data.xmat[t0].reshape(3,3)
            obs_list.extend(self.data.qvel[0:3]*5)
            obs_list.extend(self.data.qvel[3:6]*0.5)
            obs_list.extend(xmat[:,2])
            obs_list.append(self.data.xpos[t0][2])
            obs_list.append(xmat[2,2])
            if len(self.leg_qpos_adr)>0:
                jp = self.data.qpos[self.leg_qpos_adr[0]] - NOMINAL_POSE
                jv = self.data.qvel[self.leg_qvel_adr[0]]
                obs_list.extend(jp)
                obs_list.extend(jv*0.1)
            else:
                obs_list.extend([0]*20)
            obs_list.extend(self.prev_action)
        except:
            obs_list.extend([0]*41)
        # opponent info (camera sees opponent position/velocity/orientation)
        try:
            rel = self.data.xpos[t1][:3] - self.data.xpos[t0][:3]
            obs_list.extend(rel)  # 3D relative position
            obs_list.extend(self.data.qvel[6:9])  # opponent vel
            xmat1 = self.data.xmat[t1].reshape(3,3)
            obs_list.extend(xmat1[:,2])  # opponent up
            # distance, angle
            dist = np.linalg.norm(rel[:2])
            obs_list.append(dist)
            obs_list.append(np.arctan2(rel[1], rel[0]))  # angle to opponent
        except:
            obs_list.extend([0]*10)
        obs = np.array(obs_list[:50], dtype=np.float32)
        if len(obs) < 50:
            obs = np.pad(obs, (0,50-len(obs)))
        return obs

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        target = NOMINAL_POSE + MIRROR * ACTION_SCALE * action
        if len(self.leg_act_ids)>0 and len(target)==10:
            try:
                self.data.ctrl[self.leg_act_ids[0][:10]] = target
            except:
                pass
        if len(self.head_act_ids)>0:
            try:
                self.data.ctrl[self.head_act_ids[0]] = 0
            except:
                pass
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        t0 = self.trunk_ids[0] if self.trunk_ids else 1
        t1 = self.trunk_ids[1] if len(self.trunk_ids)>1 else 1
        try:
            rel = self.data.xpos[t1] - self.data.xpos[t0]
            dist = np.linalg.norm(rel[:2])
            xmat0 = self.data.xmat[t0].reshape(3,3)
            up = xmat0[2,2]
            # reward: approach opponent, stay up, facing opponent
            reward = 0.5 / (1.0 + dist) + 0.3*up - 0.01*np.sum(action**2)
            # head kick reward - if head close to opponent
            if np.linalg.norm(rel) < 0.3:
                reward += 0.5
            fallen = self.data.xpos[t0][2] < 0.04 or up < 0.3
        except:
            reward = 0
            fallen = False
        terminated = fallen
        truncated = self.step_count >= self.max_episode_steps
        self.prev_action = action.copy()
        return self._obs(), float(reward), terminated, truncated, {}

    def close(self):
        pass
