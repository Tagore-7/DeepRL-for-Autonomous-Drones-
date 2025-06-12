"""
Base_Drone_Controller.py
"""

import os
import sys
import random
import itertools
import math
import ctypes
import gymnasium as gym
import numpy as np

# import pybullet as p
import pybullet_data
import pkgutil

# import pkg_resources
import logging
from pybullet_utils import bullet_client
from importlib.resources import files
from deepRL_for_autonomous_drones.utils.Lidar import Lidar
from deepRL_for_autonomous_drones.envs.reward_functions import reward_functions
from deepRL_for_autonomous_drones.envs.drone import Drone
from deepRL_for_autonomous_drones.envs.obstacles import generateStaticTrees, loadMovingBlocks, loadStaticBlocks, loadTorusObstacles
from deepRL_for_autonomous_drones.envs.env_cfg import EnvCfg


class RedirectStream(object):
    """
    Hide some messages when building the PyBullet engine.
    """

    @staticmethod
    def _flush_c_stream(stream):
        if isinstance(stream.name, str):
            streamname = stream.name[1:-1]
            libc = ctypes.CDLL(None)
            libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()  # ensures python stream unaffected
        self.fd = open(self.file, "w+", encoding="utf-8")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, type, value, traceback):
        RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()


with RedirectStream(sys.stderr):
    import pybullet as p


class BaseDroneController(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, graphics=False):
        self.args = EnvCfg
        self.render_mode = render_mode
        self.use_graphics = graphics or (render_mode == "human")
        self._init_logger(rank=self.worker_id if hasattr(self, "worker_id") else None)
        self.logger.info("Initialized logger for this environment.")

        self._p = self._setup_client_and_physics()
        self.bullet_client_id = self._p._client

        self.use_graphics = graphics
        # ---- Parameter arguments ----#
        self.visual_mode = self.args.visual_mode
        self.launch_pad_position = self.args.launch_pad_position
        self.boundary_limits = self.args.boundary_limits
        self.target_pos = self.launch_pad_position
        self.distance_reward_weight = self.args.distance_reward_weight
        self.leg_contact_reward = self.args.leg_contact_reward
        self.gravity = self.args.gravity
        self.add_obstacles = self.args.add_obstacles
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

        self._trees_active = True
        # if self.enable_curriculum_learning:
        #     self._obstacles_active = False
        #     self._wind_effect_active = False
        #     self._static_blocks_active = False
        #     self._donut_obstacles_active = False
        #     self._moving_blocks_active = False

        #     self._wind_effect_active = False
        #     self._trees_active = False

        # wind force
        self.wind_force = np.array([0.0, 0.0, 0.0])
        self.wind_force_scale = 0.0

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
        self.EPISODE_LEN_SEC = 15
        self.CTRL_STEPS = self.EPISODE_LEN_SEC * self.CTRL_FREQ

        # ---- LIDAR settings ----#
        self.LIDAR_NUM_RAYS = 36  # Number of LIDAR rays
        self.LIDAR_MAX_DISTANCE = 10  # Max distance in meters a LIDAR ray can detect obstacles
        self.LIDAR_LINK_IDX = 4  # Index of the link from which the rays are emitted
        self.OFFSET = 0
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)

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
        self.current_raw_action = None  # Action sent by controller, possibly normalized and unclipped
        self.current_physical_action = None  # current_raw_action unnormalized if it was normalized
        self.current_clipped_action = None  # current_noisy_physical_action clipped to physical action bounds
        self.initial_reset = False
        self.at_reset = False
        self._seed = None

        self.episode_wind_active = False

        self.rng = np.random.default_rng()

        self.drone = Drone(
            rng=self.rng,
            launch_pad_position=self.launch_pad_position,
            gravity=self.gravity,
            ctrl_freq=self.CTRL_FREQ,
            bullet_client=self._p,
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

    def _init_logger(self, log_name="env_log", log_dir="env_logs", rank=None):
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_name}_{rank or os.getpid()}.log"
        log_path = os.path.join(log_dir, filename)

        self.logger = logging.getLogger(f"SafeDroneEnv-{rank or os.getpid()}")
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.propagate = False  # prevents double logging
    
    def _setup_client_and_physics(self):
        try:
            if self.use_graphics:
                bc = bullet_client.BulletClient(connection_mode=p.GUI)
                bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                bc.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            else:
                bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
        except Exception as e:
            print(f"Error creating client: {e}, falling back to DIRECT")
            bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
        
        # Verify connection
        if bc._client < 0:
            raise RuntimeError("Failed to connect to physics server")
        
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        return bc

    # def _setup_client_and_physics(self, graphics=False):
    #     with RedirectStream(sys.stdout):
    #         try:
    #             existing_connections = p.getConnectionInfo()
    #             if existing_connections and existing_connections["isConnected"] and existing_connections["connectionMethod"] == p.GUI:
    #                 print("Existing GUI connection detected. Using DIRECT to avoid conflict.")
    #                 bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
    #             else:
    #                 connection_mode = (
    #                     p.GUI
    #                     if (graphics or self.use_graphics or self.render_mode == "human" or self.args.visual_mode.upper() == "GUI")
    #                     else p.DIRECT
    #                 )
    #                 bc = bullet_client.BulletClient(connection_mode=connection_mode)
    #                 bc.configureDebugVisualizer(p.COV_ENABLE_GUI, int(connection_mode == p.GUI))
    #                 bc.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(connection_mode == p.GUI))
    #         except Exception as e:
    #             print(f"Error while setting up PyBullet client: {e}")
    #             bc = bullet_client.BulletClient(connection_mode=p.DIRECT)

    #     bc.setAdditionalSearchPath(pybullet_data.getDataPath())
    #     return bc

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
            raise RuntimeError("[ERROR] You must call env.reset() at least once before using env.step().")

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

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.logger.info("Seed: %s", seed, exc_info=1)
            self.rng = np.random.default_rng(seed)
            if hasattr(self, "drone"):
                self.drone.set_seed(seed)

    def reset(self, seed=None, options=None):
        """
        (Re-)initializes the environment to start an episode.
        Mandatory to call at least once after __init__().
        """
        # seed for reproducibility
        super().reset(seed=seed)
        self.seed(seed)
        # if seed is not None:
        #     self._seed = seed
        #     self.logger.info("Seed: %s", seed, exc_info=1)
        #     self.rng = np.random.default_rng(seed)
        #     if hasattr(self, "drone"):
        #         self.drone.set_seed(seed)

        # if not hasattr(self, "logger"):
        #     self._init_logger()

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

        info = {}

        # ---- After reset ----#
        self.after_reset()

        return obs, info

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

        # ---- Set PyBullet's parameters ----#
        self._p.resetSimulation()
        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0, 0, self.gravity)
        self._p.setTimeStep(self.PYB_TIMESTEP)

        # ---- Load ground plane, drone, launch pad, and obstacles models ----#
        self.plane = self._p.loadURDF("plane.urdf")
        self.launch_pad = self._p.loadURDF(
            str(files("deepRL_for_autonomous_drones") / "assets/launch_pad.urdf"),
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

        # ---- Calculate wind force if enabled ----#
        if self.enable_wind and self._wind_effect_active:
            # self.p_e = self.rng.uniform(0, 1)
            # self.episode_wind_active = self.p_e < 0.5

            # ---- For testing wind at various percentages ----#
            self.episode_wind_active = self.wind_force_scale > 0.0
            if self.episode_wind_active:
                print("Episode wind active")
                f_magnitude = self.rng.uniform(0, 0.005) #old  wind 
                # f_magnitude = self.rng.uniform(0.0, 1) #new  wind 
                f_direction = self.rng.uniform(-1, 1, 3)
                f_direction[2] = 0
                f_direction /= np.linalg.norm(f_direction[:2])
                self.wind_force = self.wind_force_scale * f_magnitude * f_direction
            else:
                self.wind_force = np.array([0.0, 0.0, 0.0])

        if self.use_graphics:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

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
            # rgb_obs = self._getCameraImage()
            rgb_obs = self._getCameraImage() / 255.0

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

            # rgb_obs = self._getCameraImage()
            rgb_obs = self._getCameraImage() / 255.0

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
        lidar_position = self._p.getLinkState(self.drone.getDroneID(), self.LIDAR_LINK_IDX)
        ray_from_position = [
            lidar_position[0][0],
            lidar_position[0][1],
            lidar_position[0][2],
        ]
        lidar_orientation = list(self._p.getEulerFromQuaternion(lidar_position[1]))

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
        pos, orn = self._p.getBasePositionAndOrientation(self.drone.getDroneID())
        rot = np.array(self._p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Camera position 0.1m in front of the drone
        camera_pos = pos + rot.dot([0.1, 0, 0])
        target_pos = pos + rot.dot([1, 0, 0])

        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=rot.dot([0, 0, 1]),
        )

        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far,
        )

        _, _, rgb, _, _ = self._p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=(self._p.ER_BULLET_HARDWARE_OPENGL if self.args.visual_mode.upper() == "GUI" else self._p.ER_TINY_RENDERER),
        )

        rgb_array = np.array(rgb, dtype=np.uint8)[:, :, :3]  # Remove alpha channel
        rgb_array = np.transpose(rgb_array, (2, 0, 1))  # Transpose to (C, H, W)
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array

    def _computeReward(self, observation, action, reward_function, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
        """Calls the selected reward function and computes it."""
        if reward_function not in reward_functions:
            print(f"[WARNING] Invalid reward function '{reward_function}' selected. Using default: 1")
            reward_function = 1

        return reward_functions[reward_function](self, observation, action, tilt_cost, spin_cost, lidar_cost)

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
            # rng = np.random.default_rng(seed=self._seed)

            num_trees = 50
            min_distance_from_pad = 1.0
            spawn_range = (-5, 5)

            # ---- Generate fixed tree positions with some randomness ----#
            if not hasattr(self, "fixed_tree_positions"):
                self.fixed_tree_positions = []
                attempts = 0
                while len(self.fixed_tree_positions) < num_trees and attempts < 1000:
                    x = rng.uniform(*spawn_range)
                    y = rng.uniform(*spawn_range)

                    # ---- Keep trees away from launch pad ----#
                    if np.linalg.norm([x, y]) < min_distance_from_pad:
                        attempts += 1
                        continue

                    self.fixed_tree_positions.append((x, y, 0))
                    attempts += 1

            # ---- Assign fixed tree types (random but consistent due to fixed seed) ----#
            if not hasattr(self, "fixed_tree_types"):
                self.fixed_tree_types = [tree_options[rng.integers(0, len(tree_options))] for _ in self.fixed_tree_positions]

            self.trees = generateStaticTrees(self.fixed_tree_positions, self.fixed_tree_types, self._p)

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
        self._p.resetBasePositionAndOrientation(self.first_moving_block, new_pos1, self._p.getQuaternionFromEuler([0, 0, 0]))

        # For the second moving block: oscillate along y-axis at double speed.
        new_y = amplitude * np.sin(2 * omega * current_time)
        new_pos2 = [0, new_y, 1]  # keep x=0 and fixed z = 1
        self._p.resetBasePositionAndOrientation(self.second_moving_block, new_pos2, self._p.getQuaternionFromEuler([0, 0, 0]))

    def setWindEffects(self, flag: bool):
        """Enable or diable wind effects."""
        print(f"Wind effect set to: {flag}")
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
        if hasattr(self, "_p"):
            try:
                self._p.disconnect()
            except Exception as e:
                print("[WARNING] Bullet disconnect failed:", e)

    def _showDroneLocalAxes(self):
        AXIS_LENGTH = 2 * 0.0397
        self.X_AX = self._p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[AXIS_LENGTH, 0, 0],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.drone.getDroneID(),
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.X_AX),
        )
        self.Y_AX = self._p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, AXIS_LENGTH, 0],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.drone.getDroneID(),
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Y_AX),
        )
        self.Z_AX = self._p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, AXIS_LENGTH],
            lineColorRGB=[0, 0, 1],
            parentObjectUniqueId=self.drone.getDroneID(),
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Z_AX),
        )
