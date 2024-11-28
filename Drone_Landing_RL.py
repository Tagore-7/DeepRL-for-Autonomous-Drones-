import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import shimmy
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from stable_baselines3 import PPO


class DroneLandingEnv(gym.Env):
    def __init__(self):
        super(DroneLandingEnv, self).__init__()
        
        # connect to PyBullet 
        p.connect(p.DIRECT)
        # for GUI 
        # p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # load ground plane and launch pad
        p.loadURDF("plane.urdf")
        p.loadURDF("launch_pad.urdf", [0, 0, 0], useFixedBase=True)
        self.boundary_limits = 15
        self.launch_pad_position = np.array([0, 0, 0])
        self.drone  = None
        self.launch_pad = None
        # Reward parameters
        self.reward_distance_weight = -1.0
        self.reward_velocity_weight = -0.5
        self.reward_angle_penalty = -0.3
        self.reward_leg_contact_pad = 100.0
        self.landed = False
        self.plane = p.loadURDF("plane.urdf") 


        # define action and observation space
        #self.action_space = Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)  # Example action space for x, y, z forces
        self.action_space = Box(
            low=np.array([-3, -3, -20]), 
            high=np.array([3, 3, 20]), 
            dtype=np.float32
        )
        # 3 (position) + 4 (orientation) + 3 (velocity) + 3 (angular velocity) = 13
        # position: x, y, z
        # orientation: quaternion (x, y, z, w)
        # velocity: x, y, z
        # angular velocity: x, y, z
        self.observation_space = Box(
    low=np.array([-self.boundary_limits, -self.boundary_limits, 0, -1, -1, -1, -10, -10, -10, -10, -10, -10, -10]),
    high=np.array([self.boundary_limits, self.boundary_limits, 15, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10]),
    dtype=np.float32
)


    def reset(self, seed=None):
        # seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.launch_pad = p.loadURDF("launch_pad.urdf", self.launch_pad_position, globalScaling=1.0, useFixedBase=True)
        self.drone = self.load_drone()
        self.plane = p.loadURDF("plane.urdf") 

        return self.get_observation(), {}
    
    def load_drone(self):
        # position the drone at a random height within the boundaries
        start_x = random.uniform(-self.boundary_limits + 1, self.boundary_limits - 1)
        start_y = random.uniform(-self.boundary_limits + 1, self.boundary_limits - 1)
        start_z = random.uniform(12, self.boundary_limits - 1)
        drone = p.loadURDF("cf2x.urdf", [start_x, start_y, start_z])
        return drone    

    def get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        yaw, pitch, roll = p.getEulerFromQuaternion(orn)
        return np.array([*pos, *orn, *vel, *ang_vel])
    
    def compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)

        # Compute distance to landing pad
        # distance wieght is in meters 
        # distance calculation sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
        distance_to_pad = np.linalg.norm(np.array(pos) - np.array(self.launch_pad_position))

        # Compute speed
        # speed weight is in m/s
        # speed calculation sqrt(x^2 + y^2 + z^2)
        speed = np.linalg.norm(vel)

        # Compute tilt (using the orientation quaternion)
        # tilt angle is in radians
        # tilt angle calculation abs(pitch) + abs(roll)
        # yaw value is radians [-180, 180]
        # 0: drone is facing forward in the reference frame
        # 90: drone is facing left in the reference frame
        # -90: drone is facing right in the reference frame
        # pitch value is radians [-90, 90]
        # 45: done is titled forward in the reference frame
        # -45: drone is tilted backwards in the reference frame
        # roll value is radians [-180, 180]
        # 45: drone is tilted right in the reference frame
        # -45: drone is tilted left in the reference frame
        yaw, pitch, roll = p.getEulerFromQuaternion(orn)
        # print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
        tilt_angle = abs(pitch) + abs(roll)

        # normalize the distance, tilt angle and speed
        distance_to_pad /= self.boundary_limits
        speed /= 5.0
        tilt_angle /= 1.0



        # Calculate reward
        reward = self.reward_distance_weight * distance_to_pad + self.reward_velocity_weight * speed + self.reward_angle_penalty * tilt_angle


        # bonnus reward for being close to the landing pad
        if distance_to_pad < 2 and speed < 0.1 and tilt_angle < 0.1:
            reward += 10.0
        

        # Check if the drone is in contact with the landing pad
        leg_contacts = p.getContactPoints(self.drone, self.launch_pad)
        if leg_contacts and speed < 0.1 and tilt_angle < 0.1:
            reward += self.reward_leg_contact_pad
            self.landed = True

        # altitude reward the drone should learn continously to go down descent to the landing pad slowly
        if pos[2] > 0:
            reward += 10 / pos[2]

        # high tilt angle penalty
        if tilt_angle > 0.5:
            reward -= 10.0 * (tilt_angle - 0.5)
            
        # Penalty for staying too high for too long
        if pos[2] > 10.0:
            reward -= 2.0

        # Penalty for being too far from the landing pad
        if distance_to_pad > 1.0 and speed > 0.5:
            reward -= 10.0
                    
        return reward
    
    def step(self, action):
        # apply the force as the action to the drone
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        p.applyExternalForce(self.drone, -1, action, [0, 0, 0], p.WORLD_FRAME)

        # if you want to see the drone landing simulation during training uncomment the below lines 
        # camera follows the drone 
        # pos, _ = p.getBasePositionAndOrientation(self.drone)
        # p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=pos)

        # step the simulation
        p.stepSimulation()
        time.sleep(1/240)

        # get the observation and reward and check if done 
        observation = self.get_observation()
        reward = self.compute_reward()
        terminated = self.is_done(observation) or getattr(self, "landed", False)
        truncated = False # episode is not truncated # we can use time for this 

        return observation, reward, terminated, truncated, {}
    
    def is_done(self, observation):
        pos = observation[:3]
        if (
                abs(pos[0]) > self.boundary_limits or 
                abs(pos[1]) > self.boundary_limits or 
                pos[2] > self.boundary_limits or 
                pos[2] <= 0
            ):

            return True 

        ground_contacts = p.getContactPoints(self.drone, self.plane)
        # print(f"Ground Contacts: {ground_contacts}")
        
        if any(contact[8] <= 0 for contact in ground_contacts):
            # print("Drone has hit the ground")
            return True

        if self.landed:
            return True 

        return False 
    
    def close(self):
        p.disconnect()
