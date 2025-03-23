import random
import itertools
import math
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import pkg_resources
from deepRL_for_autonomous_drones.utils.Lidar import Lidar
from deepRL_for_autonomous_drones.envs.reward_functions import reward_functions
from deepRL_for_autonomous_drones.envs.drone import Drone
from deepRL_for_autonomous_drones.envs.obstacles import (
    loadStaticBlocks,
    loadMovingBlocks,
    loadTorusObstacles,
    getFixedTreePositions,
    generateStaticTrees,
)


class BaseDroneController(gym.Env):
    def __init__(self, args):
        self.args = args
        if self.args.visual_mode.upper() == "GUI":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ---- Parameter arguments ----#
        self.visual_mode = self.args.visual_mode
        self.launch_pad_position = self.args.launch_pad_position
        self.boundary_limits = self.args.boundary_limits
        self.target_pos = self.launch_pad_position
        self.distance_reward_weight = self.args.distance_reward_weight
        self.leg_contact_reward = self.args.leg_contact_reward
        self.gravity = self.args.gravity
        self.add_obstacles = self.args.add_obstacles
        self.enable_ground_effect = self.args.enable_ground_effect
        self.enable_wind = self.args.enable_wind
        self.debug_axes = self.args.debug_axes
        self.enable_curriculum_learning = self.args.enable_curriculum_learning
        # ---- 1: Kinematic, 2: Kin+LiDAR, 3: Kin+RGB, 4: Kin+LiDAR+RGB ----#
        self.observation_type = self.args.observation_type

        # ---- Initialize curriculum-related flags (all start as OFF) ----#
        self._obstacles_active = False
        self._wind_effect_active = False
        self._static_blocks_active = False
        self._donut_obstacles_active = False
        self._moving_blocks_active = False

        self._wind_effect_active = False
        self._trees_active = True
        # if self.enable_curriculum_learning:
        #     self._obstacles_active = False
        #     self._wind_effect_active = False
        #     self._static_blocks_active = False
        #     self._donut_obstacles_active = False
        #     self._moving_blocks_active = False

        #     self._wind_effect_active = False
        #     self._trees_active = False

        # ---- Constants ----#
        self.alpha = np.array([1.0, 1.0, 1.0])
        self.beta = np.array([1.0, 1.0, 1.0])
        self.max_steps = 5000  # Maximum steps per episode
        self.urdf_path = "assets/cf2x.urdf"
        self.time_step = 1.0 / 240.0

        # ---- Set timing constants ----#
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.PYB_FREQ = 240
        self.CTRL_FREQ = 30
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1.0 / self.PYB_FREQ
        self.EPISODE_LEN_SEC = 10
        self.CTRL_STEPS = self.EPISODE_LEN_SEC * self.CTRL_FREQ

        # ---- LIDAR settings ----#
        self.LIDAR_NUM_RAYS = 36  # Number of LIDAR rays
        self.LIDAR_MAX_DISTANCE = 10  # Max distance in meters a LIDAR ray can detect obstacles
        self.LIDAR_LINK_IDX = 4  # Index of the link from which the rays are emitted
        self.OFFSET = 0
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # ---- Set RGB camera constants ----#
        self.camera_width = 64
        self.camera_height = 64
        self.camera_fov = 60
        self.camera_aspect = 1.0
        self.camera_near = 0.01
        self.camera_far = 30.0

        # ---- For debugging drone local axes ---#
        self.X_AX = -1 * np.ones(1)
        self.Y_AX = -1 * np.ones(1)
        self.Z_AX = -1 * np.ones(1)

        self.NORMALIZED_RL_ACTION_SPACE = True
        self.current_raw_action = (
            None  # Action sent by controller, possibly normalized and unclipped
        )
        self.current_physical_action = None  # current_raw_action unnormalized if it was normalized
        self.current_clipped_action = (
            None  # current_noisy_physical_action clipped to physical action bounds
        )
        self.initial_reset = False
        self.at_reset = False

        self.episode_wind_active = False

        self.rng = np.random.default_rng()

        self.drone = Drone(
            rng=self.rng,
            launch_pad_position=self.launch_pad_position,
            gravity=self.gravity,
            ctrl_freq=self.CTRL_FREQ,
            pyb_client=p,
        )

        # ---- Add observation components ----#
        self.state_obs_length = 12 + 4 * self.drone.ACTION_BUFFER_SIZE
        self.rgb_obs_shape = (3, self.camera_height, self.camera_width)

        # ---- Action Space ----#
        self.action_space = self._actionSpace()

        # ---- Observation Space ----#
        self.observation_space = self._observationSpace()

        # ---- Reset the environment ----#
        self._resetEnvironment()

        # ---- Update and store the drones kinematic information ----#
        self.drone.updateAndStoreKinematicInformation()

    def _actionSpace(self):
        """Implement in Subclasses"""
        raise NotImplementedError

    def _observationSpace(self):
        """Implement in Subclasses"""
        raise NotImplementedError

    def step(self, action):
        """Implement in Subclasses"""
        raise NotImplementedError

    def _checkInitialReset(self):
        """Makes sure that .reset() is called at least once before .step()."""
        if not self.initial_reset:
            raise RuntimeError(
                "[ERROR] You must call env.reset() at least once before using env.step()."
            )

    def before_reset(self):
        """Pre-processing before calling `.reset()`."""
        self.initial_reset = True
        self.at_reset = True
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0

        # ---- Action sent by controller, possibly normalized and unclipped ----#
        self.drone.setCurrentRawAction(None)
        # ---- Current_raw_action unnormalized if it was normalized ----#
        self.drone.setCurrentPhysicalAction(None)
        # ---- Current_noisy_physical_action clipped to physical action bounds ----#
        self.drone.setCurrentClippedAction(None)

    def after_reset(self):
        """Post-processing after calling `.reset()`."""
        self.at_reset = False

    def reset(self, seed=None, options=None):
        """
        (Re-)initializes the environment to start an episode.
        Mandatory to call at least once after __init__().
        """
        # seed for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ---- Before reset ----#
        self.before_reset()

        # ---- Reset the environment ----#
        self._resetEnvironment()

        # ---- Update and store the drones kinematic information ----#
        # self._updateAndStoreKinematicInformation()

        obs = self._getObservation()
        if self.observation_type != 1:
            if not isinstance(obs, dict):
                raise TypeError(f"Expected dict observation but got {type(obs)}: {obs}")

        # ---- After reset ----#
        self.after_reset()

        return obs, {}

    def _resetEnvironment(self):
        """
        Reset environment function.
        Allocation and zero-ing of variables and PyBullet's parameters/objects
        """
        # ---- Initialize/reset counters and zero-valued variables ----#
        self.landed = False
        self.crashed = False
        self.step_counter = 0  # Step counter for termination condition
        self.c = 0.0  # Hyperparameter indicating landing state bonus
        self.previous_shaping = None  # Previous shaping reward for temporal difference shaping
        self.last_clipped_action = np.zeros(4)

        # ---- Calculate wind force if enabled ----#
        if self.enable_wind and self._wind_effect_active:
            self.p_e = self.rng.uniform(0, 1)
            self.episode_wind_active = self.p_e < 0.8
            if self.episode_wind_active:
                f_magnitude = self.rng.uniform(0, 0.005)
                f_direction = self.rng.uniform(-1, 1, 3)
                f_direction[2] = 0
                f_direction /= np.linalg.norm(f_direction[:2])
                self.wind_force = f_magnitude * f_direction
            else:
                self.wind_force = np.array([0.0, 0.0, 0.0])

        # ---- Set PyBullet's parameters ----#
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.PYB_TIMESTEP)

        # ---- Load ground plane, drone, launch pad, and obstacles models ----#
        self.plane = p.loadURDF("plane.urdf")
        self.launch_pad = p.loadURDF(
            pkg_resources.resource_filename(
                "deepRL_for_autonomous_drones", "assets/launch_pad.urdf"
            ),
            self.launch_pad_position,
            useFixedBase=True,
        )
        self.drone.loadDrone()
        self.drone.resetDrone()

        # ---- Load obstacles if active ----#
        if self.add_obstacles:
            # self._loadStaticBlocks()
            # self._loadMovingBlocks()
            # self._loadTorusObstacles()
            self._generateStaticTrees()

        # ---- Debug local drone axes ----#
        if self.debug_axes and self.visual_mode.upper() == "GUI":
            self._showDroneLocalAxes()

    def _getObservation(self):
        """
        Returns the current observation (state) of the environment.
        # ---- [0:3] Position            ----#
        # ---- [3:7] Quaternion          ----#
        # ---- [7:10] Roll, Pitch, Yaw   ----#
        # ---- [10:13] Velocity          ----#
        # ---- [13:16] Angular Velocity  ----#
        """
        obs = self.drone.getDroneStateVector()
        obs_12 = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(
            12,
        )
        drone_state = np.array(obs_12).astype(np.float32)

        # ---- Add action buffer to observation ----#
        for i in range(self.drone.ACTION_BUFFER_SIZE):
            drone_state = np.hstack([drone_state, np.array(self.drone.action_buffer[i])])

        if np.isnan(drone_state).any():
            print("[WARNING] Found NaNs in drone state observation")
            print(drone_state)

        if self.observation_type == 2:
            lidar_results = self._getLidarSensorReadings()
            # ---- Convert each hit to a distance then normalize to [0,1] ----#
            lidar_distances = np.array(
                [res[2] * self.LIDAR_MAX_DISTANCE for res in lidar_results],
                dtype=np.float32,
            )
            normalized_lidar_dist = lidar_distances / self.LIDAR_MAX_DISTANCE
            lidar_state = normalized_lidar_dist

            return {
                "state": drone_state.astype(np.float32),
                "lidar": lidar_state.astype(np.float32),
            }
        elif self.observation_type == 3:
            rgb_obs = self._getCameraImage()

            return {
                "state": drone_state.astype(np.float32),
                "rgb": rgb_obs.astype(np.float32),
            }
        elif self.observation_type == 4:
            lidar_results = self._getLidarSensorReadings()
            # ---- Convert each hit to a distance then normalize to [0,1] ----#
            lidar_distances = np.array(
                [res[2] * self.LIDAR_MAX_DISTANCE for res in lidar_results],
                dtype=np.float32,
            )
            normalized_lidar_dist = lidar_distances / self.LIDAR_MAX_DISTANCE
            lidar_state = normalized_lidar_dist

            rgb_obs = self._getCameraImage()

            return {
                "state": drone_state.astype(np.float32),
                "lidar": lidar_state.astype(np.float32),
                "rgb": rgb_obs.astype(np.float32),
            }
        else:
            return drone_state

    def _getLidarSensorReadings(self):
        """Returns the current LiDAR sensor readings from the drone."""
        lidar = Lidar()
        lidar_position = p.getLinkState(self.drone.getDroneID(), self.LIDAR_LINK_IDX)
        ray_from_position = [
            lidar_position[0][0],
            lidar_position[0][1],
            lidar_position[0][2],
        ]
        lidar_orientation = list(p.getEulerFromQuaternion(lidar_position[1]))

        lidar_hits = lidar.CheckHits(
            ray_from_position,
            lidar_orientation,
            self.LIDAR_MAX_DISTANCE,
            self.OFFSET,
            self.launch_pad,
            draw_debug_line=self.debug_axes,
        )

        return lidar_hits

    def _getCameraImage(self):
        """Capture RGB image from drone's perspective."""
        pos, orn = p.getBasePositionAndOrientation(self.drone.getDroneID())
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Camera position 0.1m in front of the drone
        camera_pos = pos + rot.dot([0.1, 0, 0])
        target_pos = pos + rot.dot([1, 0, 0])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=rot.dot([0, 0, 1]),
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=(
                p.ER_BULLET_HARDWARE_OPENGL
                if self.args.visual_mode.upper() == "GUI"
                else p.ER_TINY_RENDERER
            ),
        )

        # process and normalize the image
        rgb_array = np.array(rgb)[:, :, :3]  # Remove alpha channel
        rgb_array = rgb_array.astype(np.float32) * 255.0  # Normalize
        rgb_array = np.transpose(rgb_array, (2, 0, 1))  # Transpose to (C, H, W)
        return rgb_array

    def _computeReward(self, observation, action, reward_function):
        """Calls the selected reward function and computes it."""
        if reward_function not in reward_functions:
            print(
                f"[WARNING] Invalid reward function '{reward_function}' selected. Using default: 1"
            )
            reward_function = 1

        return reward_functions[reward_function](self, observation, action)

    def _generateStaticTrees(self):
        self.trees = []
        if self._trees_active and self.add_obstacles:
            tree_options = [
                "assets/tree_one.urdf",
                "assets/tree_two.urdf",
                "assets/tree_three.urdf",
                "assets/tree_four.urdf",
                "assets/tree_five.urdf",
            ]

            # ---- Set a fixed random seed to ensure consistency ----#
            rng = np.random.default_rng(seed=42)

            # ---- Generate deterministic positions (evenly spaced grid) ----#
            self.fixed_tree_positions = [
                (x, y, 0)
                for x, y in itertools.product(range(-5, 6, 2), repeat=2)
                if (x, y) != (0, 0)
            ]

            # ---- Assign fixed tree types ----#
            if not hasattr(self, "fixed_tree_types"):
                self.fixed_tree_types = [
                    tree_options[rng.integers(0, len(tree_options))]
                    for _ in self.fixed_tree_positions
                ]

            self.trees = generateStaticTrees(self.fixed_tree_positions, self.fixed_tree_types)

    # def _generateStaticTrees(self):
    #     """Generates an area of trees, of various sizes, around the launch pad."""
    #     self.trees = []
    #     if self._trees_active and self.add_obstacles:
    #         tree_options = [
    #             "assets/tree_one.urdf",
    #             "assets/tree_two.urdf",
    #             "assets/tree_three.urdf",
    #             "assets/tree_four.urdf",
    #             "assets/tree_five.urdf",
    #         ]

    #         self.fixed_tree_positions = getFixedTreePositions()

    #         # ---- Select tree types for each fixed position (same for entire training) ----#
    #         if not hasattr(self, "fixed_tree_types"):
    #             self.fixed_tree_types = []
    #             for i in range(61):
    #                 if i % 5 == 0:
    #                     self.fixed_tree_types.append(tree_options[4])
    #                 elif i % 4 == 0:
    #                     self.fixed_tree_types.append(tree_options[3])
    #                 if i % 3 == 0:
    #                     self.fixed_tree_types.append(tree_options[2])
    #                 if i % 2 == 0:
    #                     self.fixed_tree_types.append(tree_options[1])
    #                 else:
    #                     self.fixed_tree_types.append(tree_options[0])

    #         self.trees = generateStaticTrees(self.fixed_tree_positions, self.fixed_tree_types)

    def _loadStaticBlocks(self):
        self.static_blocks = []
        if self._static_blocks_active and self.add_obstacles:
            self.static_blocks = loadStaticBlocks()

    def _loadMovingBlocks(self):
        self.first_moving_block = None
        self.second_moving_block = None
        if self._moving_blocks_active and self.add_obstacles:
            self.first_moving_block, self.second_moving_block = loadMovingBlocks()

    def _loadTorusObstacles(self):
        self.obstacles = []
        if self._donut_obstacles_active and self.add_obstacles:
            self.obstacles = loadTorusObstacles()

    def _updateMovingBlocks(self):
        """
        Update the positions of the moving blocks so that they oscillate along a predefined axis.
        The first moving block oscillates along the x-axis, and the second along the y-axis.
        Their movement is determined by a sine function based on simulation time.
        """
        if self.first_moving_block is None or self.second_moving_block is None:
            return

        # Define amplitude and angular frequency.
        amplitude = 3.0  # maximum displacement in meters
        omega = 0.2  # angular frequency in rad/s

        # Compute an approximate simulation time from the step counter.
        current_time = self.step_counter * self.CTRL_TIMESTEP

        # For the first moving block: oscillate along x-axis.
        new_x = amplitude * np.sin(omega * current_time)
        new_pos1 = [new_x, 0, 1]  # keep y=0 and fixed z = 1
        p.resetBasePositionAndOrientation(
            self.first_moving_block, new_pos1, p.getQuaternionFromEuler([0, 0, 0])
        )

        # For the second moving block: oscillate along y-axis at double speed.
        new_y = amplitude * np.sin(2 * omega * current_time)
        new_pos2 = [0, new_y, 1]  # keep x=0 and fixed z = 1
        p.resetBasePositionAndOrientation(
            self.second_moving_block, new_pos2, p.getQuaternionFromEuler([0, 0, 0])
        )

    def setWindEffects(self, flag: bool):
        """Enable or diable wind effects."""
        self._wind_effect_active = flag

    def setStaticBlocks(self, flag: bool):
        """Enable or disable static blocks."""
        self._static_blocks_active = flag

    def setTreesFlag(self, flag: bool):
        """Enable or disable trees."""
        self._trees_active = flag

    def setDonutObstacles(self, flag: bool):
        """Enable or disable donut obstacles."""
        self._donut_obstacles_active = flag

    def setMovingBlocks(self, flag: bool):
        """Enable or disable moving blocks."""
        self._moving_blocks_active = flag

    def close(self):
        p.disconnect()

    def _showDroneLocalAxes(self):
        AXIS_LENGTH = 2 * 0.0397
        self.X_AX = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[AXIS_LENGTH, 0, 0],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.drone,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.X_AX),
        )
        self.Y_AX = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, AXIS_LENGTH, 0],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.drone,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Y_AX),
        )
        self.Z_AX = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, AXIS_LENGTH],
            lineColorRGB=[0, 0, 1],
            parentObjectUniqueId=self.drone,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Z_AX),
        )
