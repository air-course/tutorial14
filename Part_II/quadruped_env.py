import numpy as np
import mujoco

GO2_FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]

class QuadrupedEnv:
    def __init__(
        self,
        xml_path: str,
        episode_length: int = 1000,
        frame_skip: int = 5,
        action_scale: float = 0.2,
        target_smoothing : float = 0.8,
        timestep : float = 0.002,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        self.episode_length = episode_length
        self.frame_skip = frame_skip
        self.step_count = 0

        self.action_scale = action_scale
        self.target_smoothing = target_smoothing

        # Assumption:
        # one torque actuator per actuated joint, in the same order.
        self.action_dim = self.model.nu

        self.torque_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.torque_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.actuated_joint_ids = self._get_actuated_joint_ids()
        self.actuated_qpos_ids = self._get_actuated_qpos_ids()
        self.actuated_qvel_ids = self._get_actuated_qvel_ids()

        assert len(self.actuated_qpos_ids) == self.action_dim
        assert len(self.actuated_qvel_ids) == self.action_dim

        self.default_joint_positions = self._make_default_joint_positions()
        self.q_des = self.default_joint_positions.copy()

        self.obs_dim = self._get_obs().shape[0]

        self.foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in GO2_FOOT_GEOM_NAMES
            ],
            dtype=np.int32,
        )
        
        if np.any(self.foot_geom_ids < 0):
            missing = [
                name for name, gid in zip(GO2_FOOT_GEOM_NAMES, self.foot_geom_ids)
                if gid < 0
            ]
            raise ValueError(f"Missing foot geoms in XML: {missing}")

        self.target_base_height = 0.35
        self.foot_clearance_target = 0.08
        self.base_height_weight = 8.0
        self.foot_clearance_weight = 0.5
        self.forward_velocity_weight = 10.0
        self.orientation_weight = 10.0
        self.min_upright_z = 0.5

        self.prev_foot_positions = np.zeros((4, 3), dtype=np.float32)

        self.kp = np.array([
            20, 35, 45,
            20, 35, 45,
            20, 35, 45,
            20, 35, 45,
        ], dtype=np.float32)
        
        self.kd = 0.5 * 2.0 * np.sqrt(self.kp)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        self._set_initial_configuration()

        mujoco.mj_forward(self.model, self.data)

        self.prev_foot_positions = self._get_foot_positions().copy()

        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        gait = self._gait_pattern()

        q_des_raw = self.default_joint_positions + gait + self.action_scale * action
        self.q_des = (
            self.target_smoothing * self.q_des
            + (1.0 - self.target_smoothing) * q_des_raw
        )
        q_des = self.q_des

        x_before = self.data.qpos[0].copy()

        total_torque = 0.0
        for _ in range(self.frame_skip):
            torque = self._pd_control(q_des)
            total_torque += torque
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        x_after = self.data.qpos[0].copy()

        obs = self._get_obs()
        reward = self._compute_reward(x_before, x_after, total_torque / float(self.frame_skip), q_des)
        done = self._is_done()

        dt = self.model.opt.timestep * self.frame_skip
        forward_velocity = (x_after - x_before) / dt

        info = {
            "x_position": x_after,
            "forward_velocity": forward_velocity,
            "torso_height": self.data.qpos[2],
            "energy": np.square(torque).sum(),
            "mean_abs_action": np.abs(action).mean(),
        }

        return obs, reward, done, info

    def _pd_control(self, q_des):
        q = self.data.qpos[self.actuated_qpos_ids]
        qdot = self.data.qvel[self.actuated_qvel_ids]

        torque = self.kp * (q_des - q) - self.kd * qdot
        torque = np.clip(torque, self.torque_low, self.torque_high)

        return torque

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Remove global x/y position.
        qpos_without_xy = qpos[2:]

        freq = 1.4
        t = self.data.time
        gait_phase = (freq * t) % 1.0
        phase_obs = np.array(
            [
                np.sin(2.0 * np.pi * gait_phase),
                np.cos(2.0 * np.pi * gait_phase),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([
            qpos_without_xy,
            qvel,
            phase_obs
        ])

        return obs.astype(np.float32)

    def _compute_reward(self, x_before, x_after, torque, q_des):
        dt = self.model.opt.timestep * self.frame_skip
        forward_velocity = (x_after - x_before) / dt

        torso_height = self.data.qpos[2]

        forward_reward = self.forward_velocity_weight * forward_velocity
        alive_bonus = 0.1

        torque_penalty = 0.0005 * np.square(torque).sum()

        joint_pos = self.data.qpos[self.actuated_qpos_ids]
        joint_target_penalty = 0.001 * np.square(q_des - joint_pos).sum()

        height_penalty = 0.0
        if torso_height < 0.2:
            height_penalty = 2.0

        base_z = self.data.qpos[2]
        base_height_penalty = self.base_height_weight * np.square(
            base_z - self.target_base_height
        )

        # Keep body upright.
        up_z = self._base_up_z()
        orientation_penalty = self.orientation_weight * (1.0 - up_z)

        foot_positions = self._get_foot_positions()
        foot_heights = foot_positions[:, 2]
        
        clearance_error = np.square(
            np.clip(self.foot_clearance_target - foot_heights, 0.0, None)
        )
        
        foot_clearance_reward = self.foot_clearance_weight * np.exp(
            -20.0 * clearance_error
        ).mean()

        reward = (
            forward_reward
            + alive_bonus
            + foot_clearance_reward
            - base_height_penalty
            - orientation_penalty
            - torque_penalty
            - joint_target_penalty
            - height_penalty
        )

        return float(reward)

    def _is_done(self):
        torso_height = self.data.qpos[2]
        up_z = self._base_up_z()

        fell = torso_height < 0.1
        upside_down = up_z < self.min_upright_z
        timeout = self.step_count >= self.episode_length

        return bool(fell or upside_down or timeout)

    def _get_foot_positions(self):
        return self.data.geom_xpos[self.foot_geom_ids].copy()

    def _base_up_z(self):
        """
        Returns the z component of the base local z-axis in world frame.
    
        MuJoCo free-joint quaternion convention:
        qpos[3:7] = [w, x, y, z]
    
        upright      ->  1
        sideways     ->  0
        upside down  -> -1
        """
        w, x, y, z = self.data.qpos[3:7]
    
        up_z = 1.0 - 2.0 * (x * x + y * y)
    
        return float(up_z)

    def _gait_pattern(self):
        """
        Returns joint offsets around default_joint_positions.
    
        Joint order:
        FL_hip, FL_thigh, FL_calf,
        FR_hip, FR_thigh, FR_calf,
        RL_hip, RL_thigh, RL_calf,
        RR_hip, RR_thigh, RR_calf
        """
        t = self.data.time
    
        freq = 1.4
        duty = 0.5          # 50% stance, 50% swing
        step_length = 0.25
        step_height = 0.12
    
        gait = np.zeros(12, dtype=np.float32)
    
        # Nominal foot locations in each hip frame.
        # x forward, z downward-ish in our simple sagittal approximation.
        nominal_x = {
            "front": 0.02,
            "rear": -0.02,
        }
    
        nominal_z = -0.36
    
        phases = {
            "FL": 0.0,
            "FR": 0.5,
            "RL": 0.5,
            "RR": 0.0,
        }
    
        leg_specs = [
            ("FL", 0, "front"),
            ("FR", 3, "front"),
            ("RL", 6, "rear"),
            ("RR", 9, "rear"),
        ]
    
        for leg_name, start_idx, leg_type in leg_specs:
            phase = (freq * t + phases[leg_name]) % 1.0
    
            if phase < duty:
                # Stance: foot stays low and moves backward relative to body.
                s = phase / duty
                x = nominal_x[leg_type] + step_length * (0.5 - s)
                z = nominal_z
            else:
                # Swing: foot moves forward and lifts.
                s = (phase - duty) / (1.0 - duty)
                x = nominal_x[leg_type] + step_length * (s - 0.5)
    
                # Smooth bump: 0 at start/end, max at middle.
                z = nominal_z + step_height * np.sin(np.pi * s)
    
            hip, thigh, calf = self._leg_ik_sagittal(x, z)
    
            # We use offsets from the default pose.
            gait[start_idx + 0] = hip - self.default_joint_positions[start_idx + 0]
            gait[start_idx + 1] = thigh - self.default_joint_positions[start_idx + 1]
            gait[start_idx + 2] = calf - self.default_joint_positions[start_idx + 2]
    
        return gait

    def _leg_ik_sagittal(self, x, z):
        """
        Very simple planar 2-link IK for thigh/calf.
    
        This is not a perfect Go2 kinematic model, but it gives a much better
        educational gait prior than raw sine waves.
    
        x: desired foot x position relative to hip
        z: desired foot z position relative to hip, negative downward
    
        Returns:
          hip_abduction, thigh, calf
        """
        l1 = 0.213
        l2 = 0.213
    
        d = np.sqrt(x * x + z * z)
        d = np.clip(d, 0.12, l1 + l2 - 1e-4)
    
        # Knee angle. Go2 calf joint is negative when bent.
        cos_knee = (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
    
        knee_internal = np.arccos(cos_knee)
        calf = -(np.pi - knee_internal)
    
        # Hip/thigh angle.
        # alpha = np.arctan2(x, -z)
        alpha = np.arctan2(-x, -z)
    
        cos_hip = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
        cos_hip = np.clip(cos_hip, -1.0, 1.0)
    
        beta = np.arccos(cos_hip)
    
        thigh = alpha + beta
    
        hip_abduction = 0.0
    
        return hip_abduction, thigh, calf

    def _set_initial_configuration(self):
        """
        Better initial pose:
        - body starts above the ground
        - neutral orientation
        - joints start near a standing pose
        - small noise is added to avoid overfitting to one exact state
        """

        # Floating base position: x, y, z.
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.36

        # Floating base orientation quaternion: w, x, y, z.
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        self.data.qvel[:] = 0.0

        # Standing joint configuration.
        self.data.qpos[self.actuated_qpos_ids] = self.default_joint_positions

        # # Small randomization around standing pose.
        # self.data.qpos[self.actuated_qpos_ids] += np.random.normal(
        #     loc=0.0,
        #     scale=0.02,
        #     size=self.action_dim,
        # )

        # self.data.qvel[self.actuated_qvel_ids] += np.random.normal(
        #     loc=0.0,
        #     scale=0.01,
        #     size=self.action_dim,
        # )

    def _make_default_joint_positions(self):
        """
        You should customize this for your XML.

        Common 12-DoF quadruped order:
        [
          FL_hip_abduction, FL_hip_flexion, FL_knee,
          FR_hip_abduction, FR_hip_flexion, FR_knee,
          RL_hip_abduction, RL_hip_flexion, RL_knee,
          RR_hip_abduction, RR_hip_flexion, RR_knee,
        ]

        The numbers below are only a reasonable starting point.
        """

        if self.action_dim != 12:
            return np.zeros(self.action_dim, dtype=np.float32)

        default_pose = np.array([
            0.0,  0.7, -1.2,
            0.0,  0.7, -1.2,
            0.0,  0.7, -1.2,
            0.0,  0.7, -1.2,
        ], dtype=np.float32)

        return default_pose

    def _get_actuated_joint_ids(self):
        joint_ids = []

        for actuator_id in range(self.model.nu):
            transmission_id = self.model.actuator_trnid[actuator_id, 0]
            joint_ids.append(transmission_id)

        return np.array(joint_ids, dtype=np.int32)

    def _get_actuated_qpos_ids(self):
        qpos_ids = []

        joint_ids = self._get_actuated_joint_ids()

        for joint_id in joint_ids:
            qpos_ids.append(self.model.jnt_qposadr[joint_id])

        return np.array(qpos_ids, dtype=np.int32)

    def _get_actuated_qvel_ids(self):
        qvel_ids = []

        joint_ids = self._get_actuated_joint_ids()

        for joint_id in joint_ids:
            qvel_ids.append(self.model.jnt_dofadr[joint_id])

        return np.array(qvel_ids, dtype=np.int32)
