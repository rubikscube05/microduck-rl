"""
microduck_env.py
================

A small, readable Gymnasium environment for the Microduck biped robot.

The goal of this file is to make the reinforcement-learning (RL) interface
explicit and easy to study.  Every RL environment has the same core contract:

    obs, info = env.reset(seed=...)          # start a new episode
    obs, reward, terminated, truncated, info = env.step(action)

The agent sees `obs`, chooses an `action`, and receives a scalar `reward`.
`terminated` means the task ended for a physical reason (the duck fell over).
`truncated` means we hit a time limit (the episode was cut short).

The actual physics is simulated by MuJoCo. We load the official Pollen Robotics
MJCF from `assets/scene_walk.xml` and talk to MuJoCo through its Python API.

Coordinate conventions used here
--------------------------------
* +x points forward (the direction the duck should walk).
* +y points to the duck's left.
* +z points up.
* The floating base is the `trunk_base` body with a MuJoCo `freejoint`.
* `qpos[0:3]` is the world position of the base.
* `qpos[3:7]` is the base orientation quaternion (w, x, y, z).
* The 10 leg hinge angles follow in the order of `LEG_JOINT_NAMES`.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Path to the MJCF model that was converted from the Microduck URDF.
# Keeping it relative to this file means the project can be moved anywhere.
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "scene_walk.xml"

# The 10 joints that the RL policy controls.
# Each leg has a 3-DOF hip (yaw / roll / pitch), a knee, and an ankle.
LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

# The head and jaw are not part of the walking task.  They are still actuated so
# they can be held in a neutral pose instead of flopping around.
HEAD_JOINT_NAMES = [
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
]

# A crouched standing pose found by solving for a flat-footed stance.
# Left leg:  hip yaw 0, roll 0, pitch 0, knee +1.0, ankle +1.0
# Right leg: hip yaw 0, roll 0, pitch 0, knee -1.0, ankle -1.0
# The right leg angles are mirrored because its joint axes point the other way.
# Starting from a stable crouch is much easier for PPO than starting from a
# straight-legged pose that immediately tips over.
NOMINAL_POSE = np.array(
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0],
    dtype=np.float64,
)

# The right leg's frames are mirrored.  Multiplying the right-leg actions by -1
# lets us describe a single "physical" action for both legs: e.g. positive
# knee action bends both knees the same way.
MIRROR = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.float64)

# How far (in radians) the policy may move each joint away from the crouch.
# Larger values give the policy more freedom but make exploration harder.
# Order matches LEG_JOINT_NAMES: hip yaw, hip roll, hip pitch, knee, ankle.
ACTION_SCALE = np.array([0.25, 0.25, 0.40, 0.50, 0.40, 0.25, 0.25, 0.40, 0.50, 0.40])


class MicroDuckEnv(gym.Env):
    """Gymnasium environment in which the Microduck learns to walk forward.

    Parameters
    ----------
    render_mode:
        ``None`` for headless training, ``"human"`` for the interactive MuJoCo
        viewer, or ``"rgb_array"`` to get rendered frames as numpy arrays.
    target_speed:
        Desired forward speed in metres per second.
    max_episode_steps:
        Maximum number of control steps per episode.  With ``dt = 0.01 s`` the
        default of 1000 steps is a 10-second episode.
    """

    # Gymnasium uses this metadata to describe supported render modes.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode=None,
        target_speed=0.18,
        max_episode_steps=1000,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Load the physics model and create a simulation state.
        # ------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        # MuJoCo's physics timestep is 2 ms.  We let the policy act every 5
        # physics steps, so the control frequency is 1 / (5 * 0.002) = 100 Hz.
        # Decoupling physics and control is standard: it keeps contacts stable
        # while giving the RL agent a manageable decision rate.
        self.model.opt.timestep = 0.002
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip

        self.target_speed = target_speed
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # ------------------------------------------------------------------
        # 2. Look up indices once.  MuJoCo stores everything in flat arrays,
        #    so we need the address of each joint in qpos / qvel and the id of
        #    each actuator in the ctrl array.
        # ------------------------------------------------------------------
        self.trunk_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )

        # qpos address of every leg hinge (the free joint occupies 0..6).
        self.leg_qpos_adr = np.array(
            [self.model.joint(name).qposadr[0] for name in LEG_JOINT_NAMES],
            dtype=int,
        )
        # qvel (velocity) address of every leg hinge.
        self.leg_qvel_adr = np.array(
            [self.model.joint(name).dofadr[0] for name in LEG_JOINT_NAMES],
            dtype=int,
        )

        # Actuator ids for the leg position servos.
        self.leg_act_ids = np.array(
            [self.model.actuator(name).id for name in LEG_JOINT_NAMES],
            dtype=int,
        )
        # Actuator ids for the head servos that we hold at zero.
        self.head_act_ids = np.array(
            [self.model.actuator(name).id for name in HEAD_JOINT_NAMES],
            dtype=int,
        )

        # Position actuators have a valid range (the joint limits).  We clip
        # every commanded target to this range so the servos never ask for an
        # impossible angle.
        self.ctrl_min = self.model.actuator_ctrlrange[self.leg_act_ids, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[self.leg_act_ids, 1].copy()

        # Foot collision geoms are used to place the duck on the floor at reset.
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_left_collision"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_right_collision"),
        ]

        self.n_act = len(LEG_JOINT_NAMES)

        # ------------------------------------------------------------------
        # 3. Define the RL interface.
        #
        # The observation is a flat vector of 41 numbers:
        #   3  base linear velocity (world frame, scaled)
        #   3  base angular velocity (body frame, scaled)
        #   3  gravity direction expressed in the body frame
        #   1  trunk height above the world origin
        #   1  trunk "up" component (1.0 = perfectly upright)
        #   10 leg joint angles relative to the nominal crouch
        #   10 leg joint velocities (scaled)
        #   10 previous actions (helps the policy learn smooth motion)
        # ------------------------------------------------------------------
        self.obs_dim = 3 + 3 + 3 + 2 + self.n_act + self.n_act + self.n_act
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Actions are normalized to [-1, 1].  The environment converts them to
        # joint-angle targets around the nominal crouch.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_act,), dtype=np.float32
        )

        # Bookkeeping.
        self.prev_action = np.zeros(self.n_act, dtype=np.float64)
        self.step_count = 0
        self._viewer = None
        self._renderer = None

    # ----------------------------------------------------------------------
    # Observation helpers
    # ----------------------------------------------------------------------
    def _obs(self):
        """Build the observation vector from the current MuJoCo state."""
        # `xmat` is the body-to-world rotation matrix of the trunk.
        xmat = self.data.xmat[self.trunk_id].reshape(3, 3)

        # Gravity in the body frame tells the policy which way is down.  This
        # is a compact alternative to feeding the raw quaternion.
        gravity_body = xmat.T @ np.array([0.0, 0.0, -1.0])

        # The world z-component of the body's local z-axis.  1.0 means the
        # trunk is perfectly upright; 0.0 means it is lying on its side.
        up = xmat[2, 2]

        # The free joint's linear velocity is in world coordinates, while its
        # angular velocity is in the body frame.  These are MuJoCo conventions.
        lin_vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]

        # Joint angles are expressed relative to the crouch pose so that the
        # nominal stance is near zero, which is easier for a neural network.
        joint_pos = self.data.qpos[self.leg_qpos_adr] - NOMINAL_POSE
        joint_vel = self.data.qvel[self.leg_qvel_adr]

        # The trunk height is the z position of the body in the world.
        trunk_height = self.data.xpos[self.trunk_id][2]

        # Scale some values so they are roughly O(1).  This is a simple form of
        # feature normalization.  For larger projects, SB3's VecNormalize can
        # normalize observations automatically and online.
        return np.concatenate(
            [
                lin_vel * 5.0,          # metres/s, typical values 0..0.3
                ang_vel * 0.5,          # radians/s
                gravity_body,           # already unit length
                [trunk_height, up],     # metres and a unitless cosine
                joint_pos,              # radians relative to crouch
                joint_vel * 0.1,        # radians/s, damped down
                self.prev_action,       # last normalized action
            ]
        ).astype(np.float32)

    # ----------------------------------------------------------------------
    # Floor helpers
    # ----------------------------------------------------------------------
    def _lowest_foot_z(self):
        """Return the world z of the lowest corner of the two foot boxes.

        The foot collision geoms are boxes.  For a box with half-extents
        ``size`` and world rotation ``R``, the lowest corner is:

            centre_z - |R[2, :]| dot size

        This is used only during reset to place the duck gently on the floor.
        """
        lowest = np.inf
        for gid in self.foot_geom_ids:
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            centre_z = self.data.geom_xpos[gid][2]
            half_extents = self.model.geom_size[gid]
            corner_z = centre_z - np.abs(rotation[2]).dot(half_extents)
            lowest = min(lowest, corner_z)
        return float(lowest)

    # ----------------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Start a new episode.

        ``super().reset(seed=seed)`` initializes Gymnasium's random number
        generator.  We then randomize the initial pose slightly so the policy
        does not overfit to one exact starting configuration.
        """
        super().reset(seed=seed)

        # Clear all positions, velocities, controls, and contact forces.
        mujoco.mj_resetData(self.model, self.data)

        # Small random yaw (turn) and position offsets.
        yaw = self.np_random.uniform(-0.10, 0.10)
        self.data.qpos[0] = self.np_random.uniform(-0.02, 0.02)
        self.data.qpos[1] = self.np_random.uniform(-0.01, 0.01)

        # Quaternion from a yaw rotation about the world z-axis.
        self.data.qpos[3:7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

        # Start near the crouch pose with a little joint noise.
        self.data.qpos[self.leg_qpos_adr] = NOMINAL_POSE + self.np_random.uniform(
            -0.03, 0.03, self.n_act
        )

        # Start at rest.
        self.data.qvel[:] = 0.0

        # Ask MuJoCo to compute forward kinematics so we can query foot heights.
        mujoco.mj_forward(self.model, self.data)

        # Move the base up/down until the lowest foot corner is 2 mm above the
        # floor.  A small clearance avoids a large initial contact impulse.
        self.data.qpos[2] += 0.002 - self._lowest_foot_z()
        mujoco.mj_forward(self.model, self.data)

        # Hold the head/jaw at zero.  The policy does not control them.
        self.data.ctrl[:] = 0.0

        self.prev_action[:] = 0.0
        self.step_count = 0

        return self._obs(), {}

    def step(self, action):
        """Apply one normalized action and advance the simulation by 10 ms."""
        # The policy should already output values in [-1, 1], but clipping is a
        # cheap safety net against numerical surprises.
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Convert the normalized action into joint-angle targets:
        #
        #   target = nominal pose + mirror * scale * action
        #
        # `MIRROR` flips the right leg so that a positive action has the same
        # physical meaning on both sides.
        target = NOMINAL_POSE + MIRROR * ACTION_SCALE * action
        target = np.clip(target, self.ctrl_min, self.ctrl_max)

        # Position actuators interpret `ctrl` as the desired joint angle.
        self.data.ctrl[self.leg_act_ids] = target
        self.data.ctrl[self.head_act_ids] = 0.0

        # Advance several physics steps per one control step.
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # ------------------------------------------------------------------
        # Read the quantities used by the reward function.
        # ------------------------------------------------------------------
        vx = self.data.qvel[0]          # forward speed (world x)
        vy = self.data.qvel[1]          # sideways speed (world y)
        xmat = self.data.xmat[self.trunk_id].reshape(3, 3)
        up = xmat[2, 2]                 # 1 = upright, 0 = sideways
        trunk_height = self.data.xpos[self.trunk_id][2]

        # ------------------------------------------------------------------
        # Reward shaping
        #
        # The agent only sees a scalar reward, so the design of this number is
        # the main way we communicate the task to it.
        # ------------------------------------------------------------------
        # 1. Forward progress.  This is linear in forward velocity and capped
        #    slightly above the target speed.  A linear term is important: it
        #    gives the agent a useful gradient as soon as it starts moving,
        #    whereas a narrow target-only reward can leave a standing policy
        #    stuck in a local optimum.
        forward_reward = 3.0 * np.clip(vx, 0.0, 0.25) / 0.25

        # 2. Speed tracking.  A small Gaussian centred on `target_speed`
        #    shapes the gait toward the desired speed instead of rewarding
        #    sprinting.  It is a bonus on top of the progress term.
        speed_sigma = 0.12
        speed_reward = 1.0 * np.exp(
            -((vx - self.target_speed) / speed_sigma) ** 2
        )

        # 3. Stay upright.  `up` is the cosine of the trunk tilt.
        upright_reward = 0.5 * up

        # 4. Keep the trunk near its natural crouch height.
        height_reward = 0.2 * np.exp(-((trunk_height - 0.087) / 0.02) ** 2)

        # 5. Shaping penalties: discourage sideways drift, spinning, and large
        #    or jerky controls.  Penalties are small so they do not dominate
        #    the walking objective.
        lateral_penalty = 0.5 * vy**2
        angular_penalty = 0.02 * np.sum(self.data.qvel[3:6] ** 2)
        ctrl_penalty = 0.0005 * np.sum(target**2)
        smooth_penalty = 0.02 * np.sum((action - self.prev_action) ** 2)

        # 6. A small constant bonus per step encourages survival.  Keep it much
        #    smaller than the walking rewards: a large alive bonus makes
        #    standing still an attractively safe local optimum.
        alive_bonus = 0.1

        reward = (
            forward_reward
            + speed_reward
            + upright_reward
            + height_reward
            + alive_bonus
            - lateral_penalty
            - angular_penalty
            - ctrl_penalty
            - smooth_penalty
        )

        # ------------------------------------------------------------------
        # Termination and truncation
        # ------------------------------------------------------------------
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        fallen = (trunk_height < 0.045) or (up < 0.40)
        terminated = bool(fallen or not finite)

        if terminated:
            # A larger penalty makes falling clearly worse than walking slowly.
            reward -= 10.0

        truncated = self.step_count >= self.max_episode_steps

        self.prev_action = action.copy()

        info = {
            "x_position": float(self.data.qpos[0]),
            "trunk_height": float(trunk_height),
            "up": float(up),
            "distance": float(self.data.qpos[0]),
        }

        return self._obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------------------
    # Optional rendering
    # ----------------------------------------------------------------------
    def render(self):
        """Render the current state (used by `watch.py` or for videos)."""
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def close(self):
        """Release the viewer/renderer if they were created."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
