import time
from gymnasium.spaces import Box, Dict
import numpy as np
import pybullet as p
from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController
from deepRL_for_autonomous_drones.envs.env_cfg import EnvCfg

# import pybullet_data
# import math
# import noise


class DroneControllerRPM(BaseDroneController):
    def __init__(self, render_mode=None, graphics=False):
        # super(DroneControllerRPM, self).__init__(render_mode=None, graphics=False)
        super().__init__(render_mode=render_mode, graphics=graphics)
        self.reward_function = self.args.reward_function

    def _actionSpace(self):
        """Sets the action space of the environment."""
        act_lower_bound = self.drone.KF * 1 * (self.drone.PWM2RPM_SCALE * self.drone.MIN_PWM + self.drone.PWM2RPM_CONST) ** 2
        act_upper_bound = self.drone.KF * 1 * (self.drone.PWM2RPM_SCALE * self.drone.MAX_PWM + self.drone.PWM2RPM_CONST) ** 2
        self.physical_action_bounds = (
            np.full(4, act_lower_bound, np.float32),
            np.full(4, act_upper_bound, np.float32),
        )

        # ---- Add action buffer to action space ----#
        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            self.drone.action_buffer.append(np.zeros(4, dtype=np.float32))

        if self.NORMALIZED_RL_ACTION_SPACE:
            # Normalized thrust (around hover thrust).
            self.hover_thrust = self.drone.HOVER_THRUST
            self.action_space = Box(
                low=-np.ones(4, np.float32),
                high=np.ones(4, np.float32),
                dtype=np.float32,
            )
        else:
            # Direct thrust control.
            self.action_space = Box(
                low=self.physical_action_bounds[0],
                high=self.physical_action_bounds[1],
                dtype=np.float32,
            )

        return self.action_space

    def _observationSpace(self):
        """Sets the observation space of the environment."""
        # ---- The drone state dimensions                           ----#
        # ---- [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]  ----#
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo], np.float32)
        obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi], np.float32)

        # ---- Add action buffer to observation space ----#
        # ---- Adds the drones RPM actions to the observation space            ----#
        # ---- So that the RL agent sees a recent history of actions performed ----#
        if self.NORMALIZED_RL_ACTION_SPACE:
            act_lo = [-1, -1, -1, -1]
            act_hi = [1, 1, 1, 1]
        else:
            act_lo = self.physical_action_bounds[0]
            act_hi = self.physical_action_bounds[1]

        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array(act_lo, np.float32)])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array(act_hi, np.float32)])

        # Drone state space (original drone observations + action buffer)
        state_space = Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            shape=(self.state_obs_length,),
            dtype=np.float32,
        )

        lidar_dim = self.LIDAR_NUM_RAYS
        lidar_low = np.zeros((lidar_dim,), dtype=np.float32)
        lidar_high = np.ones((lidar_dim,), dtype=np.float32)

        # ---- Kin+LiDAR ----#
        if self.observation_type == 2:
            self.observation_space = Dict(
                {
                    "state": state_space,
                    "lidar": Box(low=lidar_low, high=lidar_high, dtype=np.float32),
                }
            )
        # ---- Kin+RGB ----#
        elif self.observation_type == 3:
            self.observation_space = Dict(
                {
                    "state": state_space,
                    "rgb": Box(low=0.0, high=1.0, shape=self.rgb_obs_shape, dtype=np.float32),
                }
            )
        # ---- Kin+LiDAR+RGB ----#
        elif self.observation_type == 4:
            self.observation_space = Dict(
                {
                    "state": state_space,
                    "lidar": Box(low=lidar_low, high=lidar_high, dtype=np.float32),
                    "rgb": Box(low=0, high=255, shape=self.rgb_obs_shape, dtype=np.uint8),
                }
            )
        else:
            self.observation_space = state_space

        return self.observation_space

    def setWindScale(self, scale: float):
        self.wind_force_scale = np.clip(scale, 0.0, 1.0)

        # ---- re-apply scale to current magnitude/direction ----#
        f_dir = self.wind_force / np.linalg.norm(self.wind_force + 1e-8)
        f_mag = np.linalg.norm(self.wind_force)
        self.wind_force = self.wind_force_scale * f_mag * f_dir

    def _dragWind(self):
        """Simulates the effect of wind on the drone."""
        # _, orientation = p.getBasePositionAndOrientation(self.drone)
        # linear_vel, _ = p.getBaseVelocity(self.drone)
        # base_rot = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # relative_velocity = np.array(linear_vel) - self.wind_force

        # state = self.drone.getDroneStateVector()
        # base_rot = np.array(self._p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        # relative_velocity = np.array(state[10:13]) - self.wind_force
        # drag = np.dot(base_rot.T, self.drone.DRAG_COEFF * np.array(relative_velocity))
        # self._p.applyExternalForce(
        #     self.drone.getDroneID(),
        #     4,
        #     forceObj=drag,
        #     posObj=[0, 0, 0],
        #     flags=self._p.LINK_FRAME,
        # )

        self._p.applyExternalForce(
            self.drone.getDroneID(),
            -1,
            forceObj=self.wind_force,
            posObj=[0, 0, 0],
            flags=self._p.LINK_FRAME,
        )

    def _groundEffect(self, rpm):
        """
        Simulates ground effect, where the drone experiences increased lift
        when flying closer to the ground. It calculates additional thrust contributions
        for reach rotor. Allows for a more accurate representation of drone behavior during
        low-altitude flight.
        """

        # ---- Kin. info of all links (propellers and center of mass) ----#
        link_states = self._p.getLinkStates(
            self.drone.getDroneID(),
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1,
        )

        # ---- Simple, per-propeller ground effects ----#
        prop_heights = np.array(
            [
                link_states[0][0][2],
                link_states[1][0][2],
                link_states[2][0][2],
                link_states[3][0][2],
            ]
        )
        prop_heights = np.clip(prop_heights, self.drone.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.drone.KF * self.drone.GND_EFF_COEFF * (self.drone.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.drone.rpy[0]) < np.pi / 2 and np.abs(self.drone.rpy[1]) < np.pi / 2:
            for i in range(4):
                self._p.applyExternalForce(
                    self.drone.getDroneID(),
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=self._p.LINK_FRAME,
                )

    def beforeStep(self, action):
        """Pre-processing before calling `.step()`."""
        self._checkInitialReset()
        # Save the raw input action.
        action = np.atleast_1d(np.squeeze(action))
        if action.ndim != 1:
            raise ValueError("[ERROR]: The action returned by the controller must be 1 dimensional.")
        self.current_raw_action = action
        processed_action = self.drone.preprocessAction(action)
        return processed_action

    def afterStep(self, truncated):
        """Post-processing after calling `.step()`."""
        self.pyb_step_counter += self.PYB_STEPS_PER_CTRL
        self.ctrl_step_counter += 1

        if self.ctrl_step_counter >= self.CTRL_STEPS:
            truncated = True

        return truncated

    def _simulatePhysics(self, clipped_action):
        """Advances the environment by one simulation step."""
        # if self.PYB_STEPS_PER_CTRL > 1 and self.enable_ground_effect:
        #     self.drone.updateAndStoreKinematicInformation()
        force_is_on = self.enable_wind and self._wind_effect_active and self.episode_wind_active
        gust_active = self.rng.uniform() < 0.3

        for _ in range(self.PYB_STEPS_PER_CTRL):
            if force_is_on and gust_active:
                self._dragWind()

            self.drone.physics(clipped_action)

            position, _ = self._p.getBasePositionAndOrientation(self.drone.getDroneID())
            self._p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=position,
            )

            # if force_is_on and gust_active:
            #     self._dragWind()

            self._p.stepSimulation()

            # if self.enable_ground_effect:
            #     self._groundEffect(clipped_action)

            # if self.enable_wind and self._wind_effect_active:
            #     p_s = self.rng.uniform(0, 1)  # Probability for wind at each step
            #     if p_s < 0.3:
            #         self._dragWind()

            self.drone.last_clipped_action = clipped_action

            if self.use_graphics or self.render_mode == "human" or self.visual_mode.upper() == "GUI":
                time.sleep(self.time_step)

    def step(self, action):
        """
        Advances the environment by one control step.
        The PyBullet simulation stepped PYB_FREQ/CTRL_FREQ times.
        """
        # ---- Before step physics ----#
        rpm = self.beforeStep(action)
        clipped_action = np.reshape(rpm, 4)

        self._simulatePhysics(clipped_action)

        # ---- Update and store the drones kinematic information ----#
        self.drone.updateAndStoreKinematicInformation()

        observation = self._getObservation()
        cost, tilt_cost, spin_cost, lidar_cost = self._calculateCost(observation)
        info = {
            "cost": cost,
            "tilt_cost": tilt_cost,
            "spin_cost": spin_cost,
            "lidar_cost": lidar_cost,
        }

        reward = self._computeReward(observation, action, self.reward_function, tilt_cost, spin_cost, lidar_cost)
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        truncated = self.afterStep(truncated)

        return observation, reward, terminated, truncated, info

    def _calculateCost(self, observation):
        cost = 0.0
        lidar_cost = 0.0
        tilt_cost = 0.0
        spin_cost = 0.0

        if self.observation_type in [2, 4]:
            # ---- Lidar cost ----#
            lidar_obs_distances = observation["lidar"]
            min_distance = float(np.min(lidar_obs_distances) * self.LIDAR_MAX_DISTANCE)

            # The CF2X crazyflie is roughly 0.056m in width
            threshold = 0.1
            if min_distance < threshold:
                lidar_cost = (threshold - min_distance) / threshold
                lidar_cost = np.clip(lidar_cost, 0, 1.0)
                cost += lidar_cost

        # ---- Plane crash cost ----#
        if self._p.getContactPoints(self.drone.getDroneID(), self.plane):
            cost += 5.0
            # self.logger.info("Crashed into plane")
            self.crashed = True

        # ---- Obstacle crash cost ----#
        if self.add_obstacles:
            if any(self._p.getContactPoints(self.drone.getDroneID(), tree) for tree in self.trees):
                cost += 5.0
                # self.logger.info("Crashed into a tree")
                self.crashed = True

        obs = self.drone.getDroneStateVector()
        roll, pitch, _yaw = obs[7:10]
        wx, wy, wz = obs[13:16]
        tilt_penalty = abs(roll) + abs(pitch)
        spin_penalty = abs(wx) + abs(wy) + abs(wz)
        # tilt_cost = 0.1 * tilt_penalty
        # spin_cost = 0.1 * spin_penalty

        tilt_cost = np.clip(tilt_penalty / np.pi, 0, 1.0) * 0.5  # max ~0.5
        spin_cost = np.clip(spin_penalty / 20.0, 0, 1.0) * 0.5  # max ~0.5
        cost += tilt_cost + spin_cost

        # cost /= self.CTRL_STEPS

        # TODO: Change this so lidar_cost isn't always being returned, ie when no obstacles
        return float(cost), float(tilt_cost), float(spin_cost), float(lidar_cost)

    def _computeTerminated(self):
        """Determines if the environment is terminated or not."""
        if self.landed:
            return True

        return False

    def _computeTruncated(self):
        """Determines if the environment is truncated or not."""
        state = self.drone.getDroneStateVector()

        if (
            self.crashed
            or self._p.getContactPoints(self.drone.getDroneID(), self.plane)
            or (self.add_obstacles and any(self._p.getContactPoints(self.drone.getDroneID(), tree) for tree in self.trees))
            or (
                state[2] <= 0
                or abs(state[0]) > self.boundary_limits
                or abs(state[1]) > self.boundary_limits
                or state[1] > self.boundary_limits
            )
        ):
            self.crashed = True
            return True

        return False

    def render(self, mode="human"):
        if mode == "human":
            if not self.use_graphics:
                self._p.disconnect()
                self.use_graphics = True
                # self._p = self._setup_client_and_physics(graphics=True)
                self._p = p.connect(p.GUI)
                self.drone.set_bullet_client(self._p)
                self._resetEnvironment()
                # self.drone.set_bullet_client(self._p)
        # if mode != "rgb_array":
        #     return np.array([])
        # return
