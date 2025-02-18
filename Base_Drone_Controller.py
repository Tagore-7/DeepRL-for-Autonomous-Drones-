import argparse
import os 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import xml.etree.ElementTree as ET
import math

from stable_baselines3 import PPO, A2C
# TRPO  is in (SB3-Contrib)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class BaseDroneController(gym.Env):
    def __init__(self, args):
        self.args = args
        if self.args.visual_mode.upper() == "GUI":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.launch_pad_position = self.args.launch_pad_position
        self.boundary_limits = self.args.boundary_limits
        self.target_pos = self.launch_pad_position 
        self.distance_reward_weight = self.args.distance_reward_weight
        self.leg_contact_reward = self.args.leg_contact_reward
        self.plane = None
        self.launch_pad = None
        self.drone = None
        self.landed = False
        self.crashed = False
        self.gravity = self.args.gravity
        self.alpha = np.array([1.0, 1.0, 1.0])
        self.beta  = np.array([1.0, 1.0, 1.0])

        # self.alpha = 0.2  # Hyperparameter for az
        self.c = 0  # Hyperparameter indicating landing state bonus
        self.step_counter = 0  # Step counter for termination condition
        self.max_steps = 5000  # Maximum steps per episode
        self.previous_shaping = None  # Previous shaping reward for temporal difference shaping

        self.urdf_path = "cf2x.urdf"
        self.time_step = 1./240.
        self.total_timesteps = 2e6
        p.setGravity(0, 0, self.gravity)

        # ---- Drone physical constants ---- #
        #   MASS = 0.027 (mass of drone in kilograms)
        #   ARM = 0.0397  ("Arm length" or distance from center to rotor)
        #   THRUST2WEIGHT_RATIO = 2.25 (Ratio of maximum total thrust over the drone's weight)
        #   J = [IXX, IYY, IZZ] (Inertia matrix)
        #   J_INV = (Inverse inertia matrix)
        #   KF = 3.16e-10 (thrust coefficient - how rotor speed squared translates into thrust force.
        #   KM = 7.94e-12 (Torque (moment) coefficient - how rotor speed squared translates into rotor torque.
        #   COLLISION_H =
        #   COLLISION_R = 
        #   COLLISION_Z_OFFSET
        #   MAX_SPEED_KMH
        #   GND_EFF_COEFF = (ground effect coefficient)
        #   PROP_RADIUS =  (The physical radius of the propellers)
        #   DRAG_COEFF = [DRAG_COEFF_XY, DRAG_COEFF, XY, DRAG_COEFF_Z]
        #   DW_COEFF_1 = 
        #   DW_COEFF_2 =
        #   DW_COEFF_3 = 

        # ---- Load drone properties from the .urdf file ---- #
        self.MASS, \
        self.ARM, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] mass {:f}, arm {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.MASS, self.ARM, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))

        self.G = -self.gravity * self.MASS
        self.HOVER_RPM = np.sqrt(self.G / (4*self.KF))
        self.MAX_RPM_2 = np.sqrt((self.THRUST2WEIGHT_RATIO*self.G) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM_2**2)
        self.MAX_XY_TORQUE = (2*self.ARM*self.KF*self.MAX_RPM_2**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM_2**2)

        # ---- Action Space ---- #
        self.action_space = self._actionSpace()

        # ---- Observation Space ---- #
        self.observation_space = self._observationSpace()

        # ---- Rotor positions in URDF file ---- #
        # Should obtain directly from URDF instead
        self.rotor_positions_local = np.array([
            [ 0.028, -0.028, 0.0],  # prop0
            [-0.028, -0.028, 0.0],  # prop1
            [-0.028,  0.028, 0.0],  # prop2
            [ 0.028,  0.028, 0.0],  # prop3
        ])

        self.rng = np.random.default_rng()

        #---- Generate obstacle positions in init so they maintain position ----#
        self.obstacle_positions = []
        if self.args.add_obstacles:
            for i in range(5):
              start_x = random.uniform(-2, 2)
              start_y = random.uniform(-2, 2)
              start_z = random.uniform(1, 2)
              self.obstacle_positions.append([start_x, start_y, start_z])

    # ---- Implement in Subclasses ---- #
    def _actionSpace(self):
        raise NotImplementedError

    # ---- Implement in Subclasses ---- #
    def _observationSpace(self):
        raise NotImplementedError

    def reset(self, seed=None):
        # seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._load_world()
        return self._get_observation(), {}

    def _load_world(self):
        # reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, self.gravity)

        self.plane = p.loadURDF("plane.urdf") 
        self.launch_pad = p.loadURDF("launch_pad.urdf", self.launch_pad_position, useFixedBase=True)
        self.drone = self._load_drone()
        self.landed = False
        self.crashed = False
        self.step_counter = 0
        self.c = 0.0
        self.previous_shaping = None

        self.p_e = self.rng.uniform(0,1)
        self.wind_active = self.p_e < 0.8
        if self.wind_active:
            f_magnitude = self.rng.uniform(0, 0.005)
            f_direction = self.rng.uniform(-1, 1, 3)
            f_direction[2] = 0
            f_direction /= np.linalg.norm(f_direction[:2])
            self.wind_force = f_magnitude * f_direction
        else:
            self.wind_force = np.array([0.0, 0.0, 0.0])

        if self.args.add_obstacles:
            self.obstacles = self._load_obstacles()

        if self.args.debug_axes and self.args.visual_mode.upper() == "GUI":
            self._showDroneLocalAxes()

    def _load_drone(self):
        start_x = random.uniform(-13, 13)
        start_y = random.uniform(-13, 13)
        start_z = random.uniform(3, 13)
        # paper values
        # start_x = 0
        # start_y = 0
        # start_z = 4
        drone = p.loadURDF(str(self.urdf_path), [start_x, start_y, start_z])

        return drone
    
    def _load_obstacles(self):
        obstacle = [] * 5
        for i in range(5):
            obstacle.append(p.loadURDF(
                "box.urdf",
                self.obstacle_positions[i],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            ))
        
        return obstacle

    # ---- Implement in Subclasses ---- #
    def step(self, action):
        raise NotImplementedError

    def _get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.drone)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        observation = np.array([
            *position,
            *linear_velocity,
            roll, pitch, yaw,
            *angular_velocity
        ], dtype=np.float32)
        return observation
    
    def _first_reward_function(self, observation, action):
        """
        Computes the reward function based on the author's paper: oward end-to-end control for UAV autonomous landing via deep reinforcement learning,

        """
        px, py, pz = observation[0:3]  # Position
        vx, vy, vz = observation[3:6]  # Linear velocity
        roll, pitch, yaw = observation[6:9]
        ax, ay, az = action[0], action[1], action[2]  # Actions from the agent

        obstacle_penalty = 0

        # Compute shaping reward
        shaping = (
            -100 * np.sqrt(px**2 + py**2 + pz**2)  # Distance penalty
            - 10 * np.sqrt(vx**2 + vy**2 + vz**2)  # Velocity penalty
            -np.sqrt(ax**2 + ay**2  + az ** 2)  # Action penalty
        )

        # # Check if drone has landed safely
        # contact_points = p.getContactPoints(self.drone, self.launch_pad)
        # if contact_points and abs(vx)  < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1:
        #     self.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) # Bonus for throttle tending to zero
        #     shaping +=  self.c
        #     self.landed = True
        # elif contact_points:
        #     self.crashed = True

        contact_points = p.getContactPoints(self.drone)
        if contact_points:
            for contact in contact_points:
                drone_id = contact[1]
                contact_two_id = contact[2]
                if contact_two_id == self.launch_pad and abs(vx)  < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1:
                    print(f"First. {drone_id} made contact with: {contact_two_id}")
                    self.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) # Bonus for throttle tending to zero
                    shaping +=  self.c
                    self.landed = True
                elif any(obstacle == contact_two_id for obstacle in self.obstacles):
                    obstacle_penalty -= 50
                    self.crashed = True
                else:
                    self.crashed = True

        # Reward difference (temporal difference shaping)
        if self.previous_shaping is None:
            reward = shaping
        else:
            reward = shaping - self.previous_shaping

        self.previous_shaping = shaping

        # #---- Reward for distance to obstacles ----#
        d_max = 0.5
        eta = 0.02
        for obs_pos in self.obstacle_positions:
            d = np.linalg.norm(np.array([px, py, pz]) - obs_pos)
            if d < d_max:
                obstacle_penalty += 0.5 * eta * ((1.0/d - 1.0/d_max)**2)

        reward = reward + obstacle_penalty

        # distance_to_obstacles = [np.linalg.norm(np.array(pos[:2]) - observation[:2]) for pos in self.obstacle_positions]
        # min_distance_to_obstacle = min(distance_to_obstacles) if distance_to_obstacles else float('inf')
        # obstacle_penalty += -1 / min_distance_to_obstacle if min_distance_to_obstacle > 0 else -float('inf')
        # if min_distance_to_obstacle < 0.05:
        #     obstacle_penalty -= 50

        # reward += obstacle_penalty

        return reward
    
    def _second_reward_function(self, observation, action):
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
        current_vel = np.array(observation[3:6])
        
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
            if np.all(np.abs(current_vel) < 0.1):
                self.landed = True
                reward += 11
            else:
                self.crashed = True
        
        return reward
    
    def _third_reward_function(self, observation, action):
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
        current_vel = np.array(observation[3:6])  # Velocity (vx, vy, vz)
        roll_pitch = np.array(observation[6:8])   # Roll and pitch angles (phi, theta)

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
            if np.all(np.abs(current_vel) < 0.1):  # If the velocity is low
                self.landed = True
                reward += 10  # Small bonus for successful landing
            else:
                self.crashed = True

        return reward

    def _compute_reward(self, observation, action, reward_function):
        if reward_function == 1:
            return self._first_reward_function(observation, action)
        
        elif reward_function == 2:
            return self._second_reward_function(observation, action)
        
        elif reward_function == 3:
            return self._third_reward_function(observation, action)
        else:
            raise ValueError("Invalid reward function selected.")

    def _is_done(self, observation):
        px, py, pz = observation[0:3]

        if self.landed or self.crashed:
            return True

        # contact_with_ground = p.getContactPoints(self.drone, self.plane)
        # if contact_with_ground:
        #     self.crashed = True 
        #     return True

        contact_points = p.getContactPoints(self.drone)
        for contact in contact_points:
            contact_two_id = contact[2]
            if contact_two_id == self.plane:
                return True
            if any(obstacle == contact_two_id for obstacle in self.obstacles):
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

        tree = ET.parse(str(self.urdf_path))
        URDF_TREE = tree.getroot()

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
    