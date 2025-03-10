import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import xml.etree.ElementTree as ET
import math
import pkg_resources
from collections import deque

class BaseDroneController(gym.Env):
    def __init__(self, args):
        self.args = args
        if self.args.visual_mode.upper() == "GUI":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #---- Parameter arguments ----#
        self.launch_pad_position = self.args.launch_pad_position
        self.boundary_limits = self.args.boundary_limits
        self.target_pos = self.launch_pad_position 
        self.distance_reward_weight = self.args.distance_reward_weight
        self.leg_contact_reward = self.args.leg_contact_reward
        self.gravity = self.args.gravity
        self.add_obstacles = self.args.add_obstacles
        self.delayed_obstacles = self.args.add_obstacles
        self._obstacles_active = True

        # Initialize curriculum-related flags (all start as OFF)
        self._wind_effect_active = False
        self._static_blocks_active = False
        self._donut_obstacles_active = False
        self._moving_blocks_active = False        

        #---- Constants ----#
        self.alpha = np.array([1.0, 1.0, 1.0])
        self.beta  = np.array([1.0, 1.0, 1.0])
        self.max_steps = 5000  # Maximum steps per episode
        self.urdf_path = "assets/cf2x.urdf"
        self.time_step = 1./240.

        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.PYB_FREQ = 240
        # self.CTRL_FREQ = 240
        self.CTRL_FREQ = 30
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ

        #---- Create a buffer for the last .5 sec of actions ----#
        self.ACTION_BUFFER_SIZE = int(self.CTRL_FREQ//2)

        #---- Stores the most recent actions performed          ----#
        #---- which is added into the Observation space and     ----#
        #---- helps the RL agent learn by giving it information ----#
        #---- about what was recently commanded, enabling it to ----#
        #---- learn better stabilization and complex actions    ----#
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)

        # ---- Load drone properties from the .urdf file ---- #
        (self.MASS, #(mass of drone in kilograms) 
          self.ARM, #("Arm length" or distance from center to rotor)
          self.THRUST2WEIGHT_RATIO, #(Ratio of maximum total thrust over the drone's weight)
          self.J, #[IXX, IYY, IZZ] (Inertia matrix)
          self.J_INV, #(Inverse inertia matrix)
          self.KF, #(thrust coefficient - how rotor speed squared translates into thrust force.)
          self.KM, #(Torque (moment) coefficient - how rotor speed squared translates into rotor torque.)
          self.COLLISION_H,
          self.COLLISION_R,
          self.COLLISION_Z_OFFSET,
          self.MAX_SPEED_KMH,
          self.GND_EFF_COEFF, #(ground effect coefficient) 
          self.PROP_RADIUS, #(The physical radius of the propellers) 
          self.DRAG_COEFF, #[DRAG_COEFF_XY, DRAG_COEFF, XY, DRAG_COEFF_Z]
          self.DW_COEFF_1,
          self.DW_COEFF_2,
          self.DW_COEFF_3) = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] mass {:f}, arm {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.MASS, self.ARM, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))

        #---- Compute constants ----#
        self.G = -self.gravity * self.MASS
        self.HOVER_RPM = np.sqrt(self.G / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.G) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.ARM*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.HOVER_THRUST = 9.81 * self.MASS / 4  # Gravity compensation per motor
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

        # ---- Rotor positions in URDF file ---- #
        self.rotor_positions_local = np.array([
            [ 0.028, -0.028, 0.0],  # prop0 - Back right rotor: clockwise
            [-0.028, -0.028, 0.0],  # prop1 - Back left rotor: counterclockwise
            [-0.028,  0.028, 0.0],  # prop2 - Front left rotor: clockwise
            [ 0.028,  0.028, 0.0],  # prop3 - Front right rotor: counterclockwise
        ])

        #---- For debugging drone local axes ---#
        self.X_AX = -1*np.ones(1)
        self.Y_AX = -1*np.ones(1)
        self.Z_AX = -1*np.ones(1)

        self.rng = np.random.default_rng()

        # ---- Action Space ---- #
        self.action_space = self._actionSpace()

        # ---- Observation Space ---- #
        self.observation_space = self._observationSpace()

        #---- Reset the environment ----#
        self._resetEnvironment()

        #---- Update and store the drones kinematic information ----#
        self._updateAndStoreKinematicInformation()


    # ---- Implement in Subclasses ---- #
    def _actionSpace(self):
        raise NotImplementedError

    # ---- Implement in Subclasses ---- #
    def _observationSpace(self):
        raise NotImplementedError
    
    # ---- Implement in Subclasses ---- #
    def step(self, action):
        raise NotImplementedError

    def reset(self, seed=None):
        # seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        #---- Reset the environment ----#
        self._resetEnvironment()
        #---- Update and store the drones kinematic information ----#
        self._updateAndStoreKinematicInformation()

        return self._getObservation(), {}

    def setObstacleMode(self, mode: bool):
        """Toggle whether obstacles should be loaded on the next reset."""
        if self.delayed_obstacles:
          self._obstacles_active = mode

    def _resetEnvironment(self):
        #---- Set PyBullet's parameters ----#
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.PYB_TIMESTEP)

        #---- Load ground plane, drone, launch pad, and obstacles models ----#
        self.plane = p.loadURDF("plane.urdf") 
        self.launch_pad = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/launch_pad.urdf'), self.launch_pad_position, useFixedBase=True)

        self.drone = self._loadDrone()

        # Load static blocks if enabled
        if self._static_blocks_active:
            self.static_blocks = []
            self.static_blocks.append(p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'),
                                                basePosition=[3, 3, 3], useFixedBase=True))
            self.static_blocks.append(p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'),
                                                basePosition=[3, -3, 3], useFixedBase=True))
            self.static_blocks.append(p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'),
                                                basePosition=[-3, 3, 3], useFixedBase=True))
            self.static_blocks.append(p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'),
                                                basePosition=[-3, -3, 3], useFixedBase=True))
        else:
            self.static_blocks = []

        # Load moving blocks if enabled
        if self._moving_blocks_active:
            self.first_moving_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/moving_blocks.urdf'),
                                                basePosition=[0, 0, 1], useFixedBase=True)
            self.second_moving_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/moving_blocks.urdf'),
                                                basePosition=[0, 0, 1], useFixedBase=True)
        else:
            self.first_moving_block = None
            self.second_moving_block = None

        # Load donut obstacles if enabled
        if self._donut_obstacles_active:
            self.obstacles = self._loadObstacles()
        else:
            self.obstacles = []

        #---- Debug local drone axes ----#
        if self.args.debug_axes and self.args.visual_mode.upper() == "GUI":
            self._showDroneLocalAxes()

        #---- Initialize/reset counters and zero-valued variables ----#
        self.landed = False
        self.crashed = False
        self.step_counter = 0 # Step counter for termination condition
        self.c = 0.0 # Hyperparameter indicating landing state bonus
        self.previous_shaping = None # Previous shaping reward for temporal difference shaping
        self.last_clipped_action = np.zeros(4)

        #---- Initialize the drones kinemaatic information ----#
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)

        #---- Calculate wind force if enabled ----#
        self.p_e = self.rng.uniform(0,1)
        self._wind_effect_active = self.p_e < 0.8
        if self._wind_effect_active:
            f_magnitude = self.rng.uniform(0, 0.005)
            f_direction = self.rng.uniform(-1, 1, 3)
            f_direction[2] = 0
            f_direction /= np.linalg.norm(f_direction[:2])
            self.wind_force = f_magnitude * f_direction
        else:
            self.wind_force = np.array([0.0, 0.0, 0.0])

    def _loadDrone(self):
        start_x = random.uniform(-13, 13)
        start_y = random.uniform(-13, 13)
        start_z = random.uniform(3, 13)
        drone = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/cf2x.urdf'), [start_x, start_y, start_z])

        return drone
    
    def _loadObstacles(self):
        obstacles = []
        donut_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/torus.obj'),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        donut_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/torus.obj'),
            rgbaColor=[1, 0, 0, 1]
        )
        donut_id_one = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=donut_collision, baseVisualShapeIndex=donut_visual, basePosition=[0,0,1], baseOrientation=[1, 1, 1, 1])
        obstacles.append(donut_id_one)

        donut_id_two = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=donut_collision, baseVisualShapeIndex=donut_visual, basePosition=[0,0,2], baseOrientation=[1, 1, 1, 1])
        obstacles.append(donut_id_two)

        # add four static blocks at four corners of the launch pad 
        first_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'), basePosition=[3, 3, 3], useFixedBase=True)
        second_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'), basePosition=[3, -3, 3], useFixedBase=True)
        third_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'), basePosition=[-3, 3, 3], useFixedBase=True)
        fourth_block = p.loadURDF(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/static_blocks.urdf'), basePosition=[-3, -3, 3], useFixedBase=True)
        obstacles.extend([first_block, second_block, third_block, fourth_block])

        if self.first_moving_block is not None and self.second_moving_block is not None:
            obstacles.extend([self.first_moving_block, self.second_moving_block])


        return obstacles
    
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
        omega = 0.5      # angular frequency in rad/s

        # Compute an approximate simulation time from the step counter.
        current_time = self.step_counter * self.CTRL_TIMESTEP

        # For the first moving block: oscillate along x-axis.
        new_x = amplitude * np.sin(omega * current_time)
        new_pos1 = [new_x, 0, 1]  # keep y=0 and fixed z = 1
        p.resetBasePositionAndOrientation(self.first_moving_block, new_pos1, p.getQuaternionFromEuler([0, 0, 0]))

        # For the second moving block: oscillate along y-axis at double speed.
        new_y = amplitude * np.sin(2 * omega * current_time)
        new_pos2 = [0, new_y, 1]  # keep x=0 and fixed z = 1
        p.resetBasePositionAndOrientation(self.second_moving_block, new_pos2, p.getQuaternionFromEuler([0, 0, 0]))
    
    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.
        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """

        self.pos, self.quat = p.getBasePositionAndOrientation(self.drone)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.drone)

    def _getDroneStateVector(self):
        """Returns the state vector of the drone.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the drone.
        """
        
        state = np.hstack([self.pos, self.quat, self.rpy,
                           self.vel, self.ang_v, self.last_clipped_action])
        return state.reshape(20,)

    def _getObservation(self):
        #---- [0:3] Position            ----#
        #---- [3:7] Quaternion          ----#
        #---- [7:10] Roll, Pitch, Yaw   ----#
        #---- [10:13] Velocity          ----#
        #---- [13:16] Angular Velocity  ----#
        obs = self._getDroneStateVector()
        obs_12 = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
        ret = np.array(obs_12).astype("float32")

        #---- Add action buffer to observation ----#
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array(self.action_buffer[i])])
        return ret

    def _firstRewardFunction(self, observation, action):
        """
        Computes the reward function based on the author's paper: oward end-to-end control for UAV autonomous landing via deep reinforcement learning,

        """
        obs = self._getDroneStateVector()
        px, py, pz = obs[0:3]  # Position
        vx, vy, vz = obs[10:13]  # Linear velocity
        roll, pitch, yaw = obs[7:10]
        wx, wy, wz = obs[13:16]
        ax, ay, az, aw = action[0], action[1], action[2], action[3]  # Actions from the agent

        obstacle_penalty = 0
        plane_penalty = 0

        rel_px, rel_py, rel_pz = np.array([px, py, pz]) - self.target_pos
        rel_vx, rel_vy, rel_vz = np.array([vx, vy, vz]) - np.array([0, 0, 0])

        # Compute shaping reward
        shaping = (
            -100 * np.sqrt(rel_px**2 + rel_py**2 + rel_pz**2)  # Distance penalty
            - 10 * np.sqrt(rel_vx**2 + rel_vy**2 + rel_vz**2)  # Velocity penalty
            # -np.sqrt(ax**2 + ay**2 + az**2)  # Action penalty
            -np.sqrt(ax**2 + ay**2  + az**2 + aw**2)  # Action penalty
        )

        # Check if drone has landed safely
        contact_points = p.getContactPoints(self.drone, self.launch_pad)
        if contact_points and abs(vx)  == 0 and abs(vy) == 0 and abs(vz) == 0:
            print("Landed")
            self.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) + 10 * (1 - abs(aw)) # Bonus for throttle tending to zero
            shaping +=  self.c
            self.landed = True
        elif contact_points:
            print("Crashed")
            plane_penalty -= 5
            self.crashed = True

        contact_points_plane = p.getContactPoints(self.drone, self.plane)
        if contact_points_plane:
            plane_penalty -= 50
            self.crashed = True

        if self._static_blocks_active:
            for block in self.static_blocks:
                if p.getContactPoints(self.drone, block):
                    print("Hit static block")
                    shaping -= 25
                    self.crashed = True

        if self._donut_obstacles_active:
            for obs_obj in self.obstacles:
                if p.getContactPoints(self.drone, obs_obj):
                    print("Hit donut obstacle")
                    shaping -= 50
                    self.crashed = True 

        if self._moving_blocks_active:
            if self.first_moving_block is not None and p.getContactPoints(self.drone, self.first_moving_block):
                print("Hit moving block")
                shaping -= 50
                self.crashed = True
            if self.second_moving_block is not None and p.getContactPoints(self.drone, self.second_moving_block):
                print("Hit moving block")
                shaping -= 50
                self.crashed = True       

        # Reward difference (temporal difference shaping)
        if self.previous_shaping is None:
            reward = shaping
        else:
            reward = shaping - self.previous_shaping

        self.previous_shaping = shaping

        #---- Weighted penalty for large tilt ----#
        tilt_penalty = 0.1 * (abs(roll) + abs(pitch))

        #---- Weighted penalty for high spin ----#
        spin_penalty = 0.05 * (abs(wx) + abs(wy) + abs(wz))

        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            reward -= 1
        
        if abs(roll) < 0.5 and abs(pitch) < 0.5:
            reward += 0.5

        reward = reward + obstacle_penalty + plane_penalty - tilt_penalty - spin_penalty 
        return reward
    
    def _secondRewardFunction(self, observation, action):
        """
        Computes the reward function based on the author's paper: A reinforce-ment learning approach for autonomous control and landing of a quadrotor,
    
        reward using the exponential reward function:
        
        R(s, a) = -α^T [c1 · exp(2|e_pos|)] - β^T [c2 · exp(2|e_vel|)]
        
        where:
        e_pos = target_pos - current_pos,
        e_vel = target_vel - current_vel (assumed to be zero for landing),
        c1 = 1 - exp(-|e_pos|),
        c2 = exp(-|e_pos|).
        
        Parameters
        - observation[0:3] contains the current position.
        - observation[3:6] contains the current linear velocity.
        - self.target_pos is defined as the landing pad’s position.
        - self.alpha and self.beta are arrays (e.g., np.array([1.0, 1.0, 1.0])) tune the hyperparameters.
        """
        # Extract current state information:
        current_pos = np.array(observation[0:3])
        # velocity indices:
        current_vel = np.array(observation[6:9])
        
        # Define target state (for landing, target velocity is zero)
        target_pos = self.target_pos
        target_vel = np.array([0.0, 0.0, 0.0])
        
        # Compute errors:
        e_pos = target_pos - current_pos
        e_vel = target_vel - current_vel
        abs_e_pos = np.abs(e_pos)
        abs_e_vel = np.abs(e_vel)
        
        # Compute c1 and c2 as per the paper:
        c1 = 1 - np.exp(-abs_e_pos)
        c2 = np.exp(-abs_e_pos)
        
        # Compute the reward:
        # This implements: R(s, a) = -α^T [c1 * exp(2|e_pos|)] - β^T [c2 * exp(2|e_vel|)]
        reward = - np.dot(self.alpha, c1 * np.exp(2 * abs_e_pos)) \
                - np.dot(self.beta, c2 * np.exp(2 * abs_e_vel))
        
        contact_points = p.getContactPoints(self.drone, self.launch_pad)
        if contact_points:
            if np.all(np.abs(current_vel) == 0.0):
                self.landed = True
                reward += 100
            else:
                self.crashed = True 

        contact_points_plane = p.getContactPoints(self.drone, self.plane)
        if contact_points_plane:
            reward -= 50
            self.crashed = True

        if self._static_blocks_active:
            for block in self.static_blocks:
                if p.getContactPoints(self.drone, block):
                    print("Hit static block")
                    reward -= 25
                    self.crashed = True

        if self._donut_obstacles_active:
            for obs_obj in self.obstacles:
                if p.getContactPoints(self.drone, obs_obj):
                    print("Hit donut obstacle")
                    reward -= 50
                    self.crashed = True 

        if self._moving_blocks_active:
            if self.first_moving_block is not None and p.getContactPoints(self.drone, self.first_moving_block):
                print("Hit moving block")
                reward -= 50
                self.crashed = True
            if self.second_moving_block is not None and p.getContactPoints(self.drone, self.second_moving_block):
                print("Hit moving block")
                reward -= 50
                self.crashed = True       

        return reward
    
    def _thirdRewardFunction(self, observation, action):
        """
        Computes the reward function based on the author's paper: "Inclined Quadrotor Landing using Deep Reinforcement Learning" 
        
        r_k = -e_p - 0.2 * e_v - 0.1 * e_{phi, theta} - (0.1 * a_{phi, theta}^2) / max(e_p, 0.001)

        - e_p = || target_pos - current_pos ||  (Position error)
        - e_v = || current_vel ||              (Velocity error)
        - e_{phi, theta} = || roll_error, pitch_error || (Orientation error)
        - a_{phi, theta} = || roll_action, pitch_action || (Control effort penalty)
        
        This function is designed for **horizontal landing** on a **flat surface**.
        
        Parameters:
        - observation[0:3]: Current position [x, y, z]
        - observation[3:6]: Current velocity [vx, vy, vz]
        - observation[6:8]: Roll and Pitch angles [roll, pitch]
        - action[0:2]: Control inputs affecting roll and pitch [ax, ay]

        Returns:
        - reward (float): Computed reward value
        """
        # Extract current state information
        current_pos = np.array(observation[0:3])  # Position (x, y, z)
        current_vel = np.array(observation[6:9])  # Velocity (vx, vy, vz)
        roll_pitch = np.array(observation[3:5])   # Roll and pitch angles (phi, theta)

        # Target state (Landing pad)
        target_pos = self.target_pos  # (0,0,0) or defined landing pad

        # Compute errors
        e_p = np.linalg.norm(target_pos - current_pos)  # Position error (L2 norm)
        e_v = np.linalg.norm(current_vel)  # Velocity error (L2 norm)
        e_phi_theta = np.linalg.norm(roll_pitch)  # Orientation error (roll & pitch)

        # Control effort penalty (Action is for roll & pitch)
        a_phi_theta = np.linalg.norm(action[0:2])  # Only first two actions affect roll & pitch

        # Compute final reward
        reward = -e_p - 0.2 * e_v - 0.1 * e_phi_theta - (0.1 * (a_phi_theta**2)) / max(e_p, 0.001)

        # Check if the drone has landed safely
        contact_points = p.getContactPoints(self.drone, self.launch_pad)
        if contact_points:
            if np.all(np.abs(current_vel) == 0.0):  # If the velocity is low
                self.landed = True
                reward += 100  # Small bonus for successful landing
            else:
                self.crashed = True

        contact_points_plane = p.getContactPoints(self.drone, self.plane)
        if contact_points_plane:
            reward -= 50
            self.crashed = True

        if self._static_blocks_active:
            for block in self.static_blocks:
                if p.getContactPoints(self.drone, block):
                    print("Hit static block")
                    reward -= 25
                    self.crashed = True

        if self._donut_obstacles_active:
            for obs_obj in self.obstacles:
                if p.getContactPoints(self.drone, obs_obj):
                    print("Hit donut obstacle")
                    reward -= 50
                    self.crashed = True 

        if self._moving_blocks_active:
            if self.first_moving_block is not None and p.getContactPoints(self.drone, self.first_moving_block):
                print("Hit moving block")
                reward -= 50
                self.crashed = True
            if self.second_moving_block is not None and p.getContactPoints(self.drone, self.second_moving_block):
                print("Hit moving block")
                reward -= 50
                self.crashed = True  

        return reward

    def _computeReward(self, observation, action, reward_function):
        if reward_function == 1:
            return self._firstRewardFunction(observation, action)
        elif reward_function == 2:
            return self._secondRewardFunction(observation, action)
        elif reward_function == 3:
            return self._thirdRewardFunction(observation, action)
        else:
            raise ValueError("Invalid reward function selected.")
        
    def setWindEffects(self, flag: bool):
        """Enable or diable wind effects."""
        self._wind_effect_active = flag

    def setStaticBlocks(self, flag: bool):
        """Enable or disable static blocks."""
        self._static_blocks_active = flag

    def setDonutObstacles(self, flag: bool):
        """Enable or disable donut obstacles."""
        self._donut_obstacles_active = flag

    def setMovingBlocks(self, flag: bool):
        """Enable or disable moving blocks."""
        self._moving_blocks_active = flag
    

    def _isDone(self, observation):
        px, py, pz = observation[0:3]

        if self.crashed:
            return True

        contact_with_ground = p.getContactPoints(self.drone, self.plane)
        if contact_with_ground:
            self.crashed = True 
            return True

        if self.args.add_obstacles:
            if any(p.getContactPoints(self.drone, obstacle) for obstacle in self.obstacles):
                self.crashed = True
                return True

        if pz <= 0 or abs(px) > self.boundary_limits or abs(py) > self.boundary_limits or pz > self.boundary_limits:
            self.crashed = True 
            return True

        return False

    def close(self):
        p.disconnect()

    def _showDroneLocalAxes(self):
        AXIS_LENGTH = 2 * 0.0397
        self.X_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                  lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                  lineColorRGB=[1, 0, 0],
                                                  parentObjectUniqueId=self.drone,
                                                  parentLinkIndex=-1,
                                                  replaceItemUniqueId=int(self.X_AX),
                                                  )
        self.Y_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                  lineToXYZ=[0, AXIS_LENGTH, 0],
                                                  lineColorRGB=[0, 1, 0],
                                                  parentObjectUniqueId=self.drone,
                                                  parentLinkIndex=-1,
                                                  replaceItemUniqueId=int(self.Y_AX),
                                                  )
        self.Z_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                  lineToXYZ=[0, 0, AXIS_LENGTH],
                                                  lineColorRGB=[0, 0, 1],
                                                  parentObjectUniqueId=self.drone,
                                                  parentLinkIndex=-1,
                                                  replaceItemUniqueId=int(self.Z_AX),
                                                  )

    # ---- Parser for CF2X.URDF file ---- #
    def _parseURDFParameters(self):
        """Loads parameters from a URDF file."""

        URDF_TREE = ET.parse(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'assets/cf2x.urdf')).getroot()

        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
                GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
            destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
            direction / distance
        )  # Normalize the direction vector
        next_step = (
            current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
    