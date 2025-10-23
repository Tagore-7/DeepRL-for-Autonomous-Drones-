import os
import sys
import random
import itertools
import math
import ctypes
import gymnasium as gym
import numpy as np

import pybullet_data
import pkgutil

import logging
from pybullet_utils import bullet_client
from importlib.resources import files
from deepRL_for_autonomous_drones.utils.Lidar import Lidar
from deepRL_for_autonomous_drones.envs.reward_functions import reward_functions
from deepRL_for_autonomous_drones.envs.cost_functions import cost_functions
from deepRL_for_autonomous_drones.envs.drone import Drone
from deepRL_for_autonomous_drones.envs.obstacles import generateStaticTrees
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
        # self._init_logger(rank=self.worker_id if hasattr(self, "worker_id") else None)
        # self.logger.info("Initialized logger for this environment.")

        self._p = self._setup_client_and_physics()
        self.bullet_client_id = self._p._client

        self.max_episode_steps = 10000
        self.normal_landing_count = 0
        self.soft_landing_count = 0

        self.use_graphics = graphics
        # ---- Parameter arguments ----#
        self.visual_mode = self.args.visual_mode
        self.landing_pad_position = self.args.launch_pad_position
        self.boundary_limits = self.args.boundary_limits
        self.target_pos = self.landing_pad_position
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
        self._wind_effect_active = True
        self._trees_active = True

        # ------- Seed / layout_pool -------#
        self.episode_idx = 0
        self.train_layout_seeds = None  # filled once on first reset
        self.eval_layout_seeds = None
        self.layout_pool_size = self.args.layout_pool_size  # number of distinct layouts for training
        self.eval_pool_size = self.args.eval_pool_size  # held-out layouts
        self.use_layout_pool = self.args.use_layout_pool

        if self.enable_curriculum_learning:
            self._obstacles_active = False
            self._wind_effect_active = False
            self._trees_active = False

        # wind force
        self.wind_force = np.array([0.0, 0.0, 0.0])
        self.wind_force_scale = 0.0
        self.wind_magnitude: float = 0
        self.episode_wind_active = False
        self.current_wind_level = "none"
        self.WIND_LEVELS = {
            "none": 0.0,
            "light_breeze": 2.24,  # 5 mph
            "light": 4.47,  # 10 mph
            "medium": 8.94,  # 20 mph
            "high": 17.88,  # 40 mph
        }

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
        self.EPISODE_LEN_SEC = 20
        self.CTRL_STEPS = self.EPISODE_LEN_SEC * self.CTRL_FREQ
        self.WIND_DELAY_STEPS = 20

        # ---- LIDAR settings ----#
        self.LIDAR_NUM_RAYS = 144  # Number of LIDAR rays
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

        self.rng = np.random.default_rng()

        self.drone = Drone(
            rng=self.rng,
            landing_pad_position=self.landing_pad_position,
            gravity=self.gravity,
            ctrl_freq=self.CTRL_FREQ,
            bullet_client=self._p,
        )

        # ---- Add observation components ----#

        # ------ landing_pad ------#
        if self.args.use_dyn_landing_pad:
            self.state_obs_length = 12 + 4 * self.drone.ACTION_BUFFER_SIZE + 3
        else:
            self.state_obs_length = 12 + 4 * self.drone.ACTION_BUFFER_SIZE
        self.rgb_obs_shape = (3, self.camera_height, self.camera_width)

        # ---- Action Space ----#
        self.action_space = self._actionSpace()

        # ---- Observation Space ----#
        self.observation_space = self._observationSpace()

        # ---- Reset the environment ----#
        # self._resetEnvironment()

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

    def _setup_client_and_physics(self, graphics=False):
        with RedirectStream(sys.stdout):
            try:
                existing_connections = p.getConnectionInfo()
                if existing_connections and existing_connections["isConnected"] and existing_connections["connectionMethod"] == p.GUI:
                    print("Existing GUI connection detected. Using DIRECT to avoid conflict.")
                    bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
                else:
                    connection_mode = (
                        p.GUI
                        if (graphics or self.use_graphics or self.render_mode == "human" or self.args.visual_mode.upper() == "GUI")
                        else p.DIRECT
                    )
                    bc = bullet_client.BulletClient(connection_mode=connection_mode)
                    bc.configureDebugVisualizer(p.COV_ENABLE_GUI, int(connection_mode == p.GUI))
                    bc.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(connection_mode == p.GUI))
            except Exception as e:
                print(f"Error while setting up PyBullet client: {e}")
                bc = bullet_client.BulletClient(connection_mode=p.DIRECT)

        # bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        return bc

    def getBulletClient(self):
        return self._p

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

        self.action_err_sum = 0.0
        self.action_err_steps = 0

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
            # self.logger.info("Seed: %s", seed, exc_info=1)
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
        # self.seed(seed)
        # if seed is not None:
        #     self._seed = seed
        #     self.logger.info("Seed: %s", seed, exc_info=1)
        #     self.rng = np.random.default_rng(seed)
        #     if hasattr(self, "drone"):
        #         self.drone.set_seed(seed)

        # if not hasattr(self, "logger"):
        #     self._init_logger()

        if seed is not None:
            self._seed = seed
            self.drone.rng = np.random.default_rng(seed)

        # ------- Seed / layout_pool -------#
        if self.args.use_dyn_trees:
            if self.use_layout_pool and self.train_layout_seeds is None:
                all_draws = self.np_random.integers(0, 2**31 - 1, size=(self.layout_pool_size + self.eval_pool_size,), dtype=np.int64)
                self.train_layout_seeds = all_draws[: self.layout_pool_size].tolist()
                self.eval_layout_seeds = all_draws[self.layout_pool_size :].tolist()

            if self.use_layout_pool:
                mix_p = 0.10
                if self.np_random.random() < mix_p:
                    layout_seed = int(self.np_random.integers(0, 2**31 - 1))
                else:
                    layout_seed = int(self.train_layout_seeds[self.episode_idx % len(self.train_layout_seeds)])
            else:
                layout_seed = int(self.np_random.integers(0, 2**31 - 1))

            self._current_layout_seed = layout_seed

        # ---- Before reset ----#
        self.before_reset()

        # ---- Reset the environment ----#
        self._resetEnvironment()

        # ---- Update and store the drones kinematic information ----#
        self.drone.updateAndStoreKinematicInformation()

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
        self.hard_landing = False
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

        # ------ landing_pad ------#
        self._add_origin_marker(height=2.0, radius=0.03)

        # ---- Load ground plane, drone, launch pad, and obstacles models ----#
        self.plane = self._p.loadURDF("plane.urdf")

        # ------ landing_pad ------#
        if self.args.use_dyn_landing_pad:
            self.landing_pad_position = self._sample_landing_pad_spawn()
        self.landing_pad = self._p.loadURDF(
            str(files("deepRL_for_autonomous_drones") / "assets/launch_pad.urdf"),
            self.landing_pad_position,
            useFixedBase=True,
        )

        self.launch_pad_position = self._sample_launch_pad_spawn()
        self.launch_pad = self._p.loadURDF(
            str(files("deepRL_for_autonomous_drones") / "assets/launch_pad.urdf"),
            self.launch_pad_position,
            useFixedBase=True,
        )

        spawn_pos = [self.launch_pad_position[0], self.launch_pad_position[1], self.launch_pad_position[2] + 0.2]
        self.drone.loadDrone(start_pos=spawn_pos)
        self.drone.action_buffer.clear()
        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            self.drone.action_buffer.append(np.zeros(4, dtype=np.float32))
        self.drone.resetDrone()
        # ------ landing_pad ------#
        self.drone.setLandingPadPosition(self.landing_pad_position)

        # ---- Load obstacles if active ----#
        if self.add_obstacles:
            if self.args.use_dyn_trees:
                self._generateStaticTrees(self._current_layout_seed)
            else:
                self._generateStaticTrees()

        # ---- Debug local drone axes ----#
        if self.debug_axes and self.visual_mode.upper() == "GUI":
            self._showDroneLocalAxes()

        if self.enable_wind:
            self._initWind()

        if self.use_graphics:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    # ------ landing_pad ------#
    def _sample_landing_pad_spawn(self):
        """Sample a (x,y,0) inside a 1.0 m radius clearing centered at the origin."""
        R = 1.0  # radius (meters)
        # Uniform over disc: r = sqrt(u) * R, theta ~ U[0, 2pi]
        u = float(self.np_random.random())
        r = np.sqrt(u) * R
        theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        x = float(r * np.cos(theta))
        y = float(r * np.sin(theta))
        z = 0.0
        return (x, y, z)

    def _sample_launch_pad_spawn(self):
        lo, hi = -10.0, 10.0

        forest_r = 5
        min_r = forest_r + 2.0  # 1m margin beyond trees
        max_r = hi - 0.5  # keep a small buffer from the edge

        # If min_r is already too big, clamp
        if min_r >= max_r:
            min_r = forest_r + 0.5
            max_r = hi - 0.5

        # for _ in range(200):
        while True:
            # Sample radius uniformly in annulus and angle uniformly in [0, 2π)
            r = self.rng.uniform(min_r, max_r)
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            if not (lo <= x <= hi and lo <= y <= hi):
                continue
            return (float(x), float(y), 0.0)

    def _add_origin_marker(self, height: float = 2.0, radius: float = 0.03):
        """
        Adds a transparent, non-colliding vertical marker at the world origin.
        """
        # Visual-only cylinder (no collision)
        vis = self._p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[1.0, 0.0, 0.0, 0.25],
            visualFramePosition=[0, 0, height / 2.0],
        )
        self._origin_marker_id = self._p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0, 1],
        )
        try:
            self._p.setCollisionFilterGroupMask(self._origin_marker_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        except Exception:
            pass

    def setWindLevel(self, level: str):
        self.current_wind_level = level

    def _initWind(self):
        self.episode_wind_active = True
        # self.p_e = self.rng.uniform(0, 1)
        # self.episode_wind_active = self.p_e < 0.7

        self._wind_direction = self.rng.uniform(-1, 1, 3)
        self._wind_direction[2] = 0
        self._wind_direction /= np.linalg.norm(self._wind_direction[:2]) + 1e-8

    def _applyWindDrag(self):
        """
        https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/drageq.html

        Drag force formula:
        F_drag = C_d * A * .5 * r * V^2

        F_drag: drag force (Newtons)
        C_d: drag coefficient
        A: reference area (The frontal area of the drone being enacted upon)
        r: air density (Typically 1.225 Kg/m^3 at sea level)
        v: relative wind velocity with respect to the drone

        A and C_d for drone taken from: https://github.com/jjshoots/PyFlyt/blob/master/PyFlyt/models/vehicles/cf2x/cf2x.yaml
        """

        if not (self.enable_wind and self._wind_effect_active):
            return

        wind_speed = self.WIND_LEVELS[self.current_wind_level]
        if wind_speed == 0.0:
            return

        wind_velocity = wind_speed * self._wind_direction

        state = self.drone.getDroneStateVector()
        px, py, pz = state[0:3]  # Drone position
        drone_velocity = np.array(state[10:13])

        vel_rel = wind_velocity - drone_velocity
        vel_mag = np.linalg.norm(vel_rel)

        r_density = 1.225
        C_d = 3.0
        A_ref = 4e-4

        F_drag = 0.5 * r_density * C_d * A_ref * vel_mag * vel_rel
        self._p.applyExternalForce(self.drone.getDroneID(), -1, forceObj=F_drag, posObj=[px, py, pz], flags=self._p.WORLD_FRAME)

        # ---- Only printing at certain steps to not spam console ----#
        # if self.pyb_step_counter % 240 == 0:
        #     print(f"|F|={np.linalg.norm(F_drag):.3f} N  " f"level={self.current_wind_level}")

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

        # ------ landing_pad ------#
        if self.args.use_dyn_landing_pad:
            drone_state = np.hstack([drone_state, np.array(self.landing_pad_position, dtype=np.float32)])

        assert not np.isnan(drone_state).any()
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
            ray_from_position=ray_from_position,
            ray_orientation=lidar_orientation,
            ray_length=self.LIDAR_MAX_DISTANCE,
            offset=self.OFFSET,
            landing_pad_id=self.landing_pad,
            launch_pad_id=self.launch_pad,
            plane=self.plane,
            draw_debug_line=self.debug_axes,
            pyb_client=self._p,
            drone_id=self.drone.getDroneID(),
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

    def _computeCost(self, observation, cost_function):
        """Calls the selected cost function and computes it."""
        if cost_function not in cost_functions:
            print(f"[WARNING] Invalid cost function '{cost_function}' selected. Using default: 1")
            cost_function = 1

        return cost_functions[cost_function](self, observation)

    def _generateStaticTrees(self, layout_seed: int = 42):
        """
        Spawns static trees reproducibly.
        Keeps: self.trees, self.fixed_tree_positions, self.fixed_tree_types (URDF only)
        """
        self.trees = []
        if not getattr(self, "_trees_active", True) or not getattr(self, "add_obstacles", True):
            return

        if self.args.use_dyn_trees:
            rng = np.random.default_rng(int(layout_seed))
        else:
            rng = np.random.default_rng(seed=42)

        min_x, max_x = -self.args.forest_span, self.args.forest_span
        min_y, max_y = -self.args.forest_span, self.args.forest_span

        launch_xy = self.launch_pad_position[:2]
        landing_xy = self.landing_pad_position[:2]

        # ------------------- Rejection-sample positions -------------------- #
        positions = []
        attempts, max_attempts = 0, max(1000, 50 * self.args.num_trees)

        def ok_xy(xy: np.ndarray) -> bool:
            if np.linalg.norm(xy - launch_xy) < self.args.launch_pad_clearance:
                return False
            if np.linalg.norm(xy - landing_xy) < self.args.landing_pad_clearance:
                return False
            if not (min_x <= xy[0] <= max_x and min_y <= xy[1] <= max_y):
                return False
            if self.args.min_tree_spacing > 0.0 and positions:
                d2_min = min(((xy[0] - px) ** 2 + (xy[1] - py) ** 2) for (px, py, _pz) in positions)
                if d2_min < (self.args.min_tree_spacing**2):
                    return False
            return True

        while len(positions) < self.args.num_trees and attempts < max_attempts:
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if ok_xy(np.array([x, y], dtype=np.float32)):
                positions.append((float(x), float(y), 0.0))
            attempts += 1

        self.fixed_tree_positions = positions

        # -------------------------- Spawn trees ----------------------------
        tree_options = [
            # "assets/tree_one.urdf",
            # "assets/tree_two.urdf",
            # "assets/tree_three.urdf",
            # "assets/tree_four.urdf",
            # "assets/tree_five.urdf",
            "assets/tree_dynamic.urdf"
        ]
        self.fixed_tree_types = [tree_options[int(rng.integers(0, len(tree_options)))] for _ in positions]
        self.trees = generateStaticTrees(self.fixed_tree_positions, self.fixed_tree_types, self._p)

    def setWindEffects(self, flag: bool):
        """Enable or diable wind effects."""
        self._wind_effect_active = flag

    def enableWind(self, flag: bool):
        self.enable_wind = flag

    def setTreesFlag(self, flag: bool):
        """Enable or disable trees."""
        self._trees_active = flag

    def enableLidarPenalty(self, flag: bool):
        self.lidar_added = flag

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
