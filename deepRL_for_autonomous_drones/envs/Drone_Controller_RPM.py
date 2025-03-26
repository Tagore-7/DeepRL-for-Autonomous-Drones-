import time
from gymnasium.spaces import Box, Dict
import numpy as np
import pybullet as p
from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController

# import pybullet_data
# import math
# import noise


class DroneControllerRPM(BaseDroneController):
    def __init__(self, args):
        super(DroneControllerRPM, self).__init__(args=args)
        self.reward_function = self.args.reward_function

    def _actionSpace(self):
        """Sets the action space of the environment."""
        act_lower_bound = (
            self.drone.KF
            * 1
            * (self.drone.PWM2RPM_SCALE * self.drone.MIN_PWM + self.drone.PWM2RPM_CONST) ** 2
        )
        act_upper_bound = (
            self.drone.KF
            * 1
            * (self.drone.PWM2RPM_SCALE * self.drone.MAX_PWM + self.drone.PWM2RPM_CONST) ** 2
        )
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

    def _dragWind(self):
        """Simulates the effect of wind on the drone."""
        # _, orientation = p.getBasePositionAndOrientation(self.drone)
        # linear_vel, _ = p.getBaseVelocity(self.drone)
        # base_rot = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # relative_velocity = np.array(linear_vel) - self.wind_force

        state = self.drone.getDroneStateVector()
        base_rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        relative_velocity = np.array(state[10:13]) - self.wind_force

        drag = np.dot(base_rot.T, self.drone.DRAG_COEFF * np.array(relative_velocity))
        p.applyExternalForce(
            self.drone.getDroneID(),
            4,
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )

    def _groundEffect(self, rpm):
        """
        Simulates ground effect, where the drone experiences increased lift
        when flying closer to the ground. It calculates additional thrust contributions
        for reach rotor. Allows for a more accurate representation of drone behavior during
        low-altitude flight.
        """

        # ---- Kin. info of all links (propellers and center of mass) ----#
        link_states = p.getLinkStates(
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
        gnd_effects = (
            np.array(rpm**2)
            * self.drone.KF
            * self.drone.GND_EFF_COEFF
            * (self.drone.PROP_RADIUS / (4 * prop_heights)) ** 2
        )
        if np.abs(self.drone.rpy[0]) < np.pi / 2 and np.abs(self.drone.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(
                    self.drone.getDroneID(),
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                )

    def beforeStep(self, action):
        """Pre-processing before calling `.step()`."""
        self._checkInitialReset()
        # Save the raw input action.
        action = np.atleast_1d(np.squeeze(action))
        if action.ndim != 1:
            raise ValueError(
                "[ERROR]: The action returned by the controller must be 1 dimensional."
            )
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
        if self.PYB_STEPS_PER_CTRL > 1 and self.enable_ground_effect:
            self.drone.updateAndStoreKinematicInformation()
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self.drone.physics(p, clipped_action)

            position, _ = p.getBasePositionAndOrientation(self.drone.getDroneID())
            p.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=position,
            )
            p.stepSimulation()

            if self.enable_ground_effect:
                self._groundEffect(clipped_action)

            if self.enable_wind and self._wind_effect_active:
                p_s = self.rng.uniform(0, 1)  # Probability for wind at each step
                if p_s < 0.3:
                    self._dragWind()

            self.drone.last_clipped_action = clipped_action

            if self.visual_mode.upper() == "GUI":
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
        reward = self._computeReward(observation, action, self.reward_function)
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        truncated = self.afterStep(truncated)

        return observation, reward, terminated, truncated, {}

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
            or p.getContactPoints(self.drone.getDroneID(), self.plane)
            or (
                self.add_obstacles
                and any(p.getContactPoints(self.drone.getDroneID(), tree) for tree in self.trees)
            )
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

    def render(self):
        return
