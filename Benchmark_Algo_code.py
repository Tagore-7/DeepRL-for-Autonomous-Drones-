# Python file for benchmarking the DeepRL algorithms by using Pybullet environment (openai gym) tensorboard and stable baselines3 library. 
# for the drone landing task.

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
from stable_baselines3 import PPO, A2C
# TRPO  is in (SB3-Contrib)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking DeepRL algorithms for drone landing task')
    parser.add_argument(
        '--algorithm_name',
        type = str,
        default= 'PPO',
        help = 'The name of the algorithm to benchmark. Options: PPO, A2C, TRPO',
    )
    parser.add_argument(
        '--boundary_limits',
        type = int,
        default = 20,
        help = 'The boundary limits for the drone to fly in the environment',
    )
    parser.add_argument(
        '--launch_pad_position',
        type = lambda x: np.array([float(i) for i in x.split(",")]),
        default = np.array([0, 0, 0]),
        help = 'The position of the launch pad for the drone to land on in the environment',
    )
    parser.add_argument(
        '--gravity',
        type = float,
        default = -9.8,
        help = 'The gravity value for the environment',
    )
    parser.add_argument(
        '--distance_reward_weight',
        type = float,
        default = 2.0,
        help = 'The weight for the distance reward (distance between drone and launch pad) in the environment',
    )
    parser.add_argument(
        '--leg_contact_reward',
        type = int,
        default = 100,
        help = 'The reward for the drone making contact with the launch pad',
    )
    parser.add_argument(
        '--tensorboard_log_dir',
        type = str,
        default = './logs_metrics_benchmark_tensorboard/',
        help = 'The directory to store the tensorboard logs for the training', 
    )
    parser.add_argument(
        '--model_name_to_save',
        type = str,
        default = 'drone_landing_model_using_ppo',
        help = 'The name of the model to save after training',
    )
    parser.add_argument(
        '--visual_mode',
        type = str,
        default = "DIRECT",
        help = 'visual mode of the environment GUI or DIRECT',
    )
    



    return parser.parse_args()

args = parse_args()

class DroneLandingEnv(gym.Env):
    def __init__(self):
        super(DroneLandingEnv, self).__init__()
        
        if args.visual_mode.upper() == "GUI":
            # for  debussing and better visualization, use GUI mode
            p.connect(p.GUI)
        else:
            # for faster training, use DIRECT mode
            p.connect(p.DIRECT)
        # p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, args.gravity)
        # p.loadURDF("plane.urdf")
        # p.loadURDF("launch_pad.urdf", args.launch_pad_position, globalScaling=1.0, useFixedBase=True)
        self.plane = None
        self.launch_pad = None
        self.launch_pad_position = args.launch_pad_position
        self.boundary_limits = args.boundary_limits
        self.drone = None
        self.distance_reward_weight = args.distance_reward_weight
        self.leg_contact_reward = args.leg_contact_reward
        self.landed = False
        self.crashed = False


        # action space
        self.action_space = Box(
            low=np.array([-3, -3, -10]), 
            high=np.array([3, 3, 10]), 
            dtype=np.float32
        )

        # observation space
        self.observation_space = Box(
    low=np.array([-self.boundary_limits, -self.boundary_limits, 0, -1, -1, -1, -10, -10, -10, -10, -10, -10, -10]),
    high=np.array([self.boundary_limits, self.boundary_limits, 20, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10]),
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
        self.plane = p.loadURDF("plane.urdf") 
        # globalScaling is a parameter that scales the size of the object in the environment
        self.launch_pad = p.loadURDF("launch_pad.urdf", self.launch_pad_position, globalScaling=1.0, useFixedBase=True)
        self.drone = self.load_drone()
        self.landed = False
        self.crashed = False

        return self.get_observation(), {}

    def load_drone(self):
        start_x = random.uniform(-15, 15)
        start_y = random.uniform(-15, 15)
        start_z = random.uniform(5, 15)
        drone = p.loadURDF("cf2x.urdf", [start_x, start_y, start_z])
        return drone 
    
    def get_observation(self):
        drone_position, drone_orientation = p.getBasePositionAndOrientation(self.drone)
        drone_linear_velocity, drone_angular_velocity = p.getBaseVelocity(self.drone)
        drone_yaw, drone_pitch, drone_roll = p.getEulerFromQuaternion(drone_orientation)

        return np.array([*drone_position, *drone_orientation, *drone_linear_velocity, *drone_angular_velocity])
    
    def compute_reward(self, observation):
        reward = 0
        drone_position = observation[:3]
        drone_orientation = observation[3:7]
        drone_linear_velocity = observation[7:10]
        drone_angular_velocity = observation[10:]

        drone_distance_from_launch_pad = np.linalg.norm(np.array(drone_position) - np.array(self.launch_pad_position))

        epsilon = 0.1
        c = 10 
        raw_dist_reward = c / (drone_distance_from_launch_pad + epsilon)
        dist_reward = min(raw_dist_reward, 10.0) # clip the reward to 10.0 to aviod large rewards when the drone is close to the pad to stop it from being hovering 
        reward += self.distance_reward_weight * dist_reward

        # Reward for reducing altitude (but not going below pad level):
        desired_pad_altitude = 0.1  # ~ top of pad
        altitude = drone_position[2]
        if altitude > desired_pad_altitude:
            # A small negative penalty that grows with altitude
            # to encourage descending
            reward -= 0.01 * (altitude - desired_pad_altitude)



        # penalize the drone for flying out of boundary limits
        if abs(observation[0]) > self.boundary_limits or abs(observation[1]) > self.boundary_limits or observation[2] > self.boundary_limits:
            reward -= 50
            self.crashed = True
            print("Drone flew out of boundary limits")


        contact_with_pad = p.getContactPoints(self.drone, self.launch_pad)
        if contact_with_pad:
            # loop thorugh each contact point 
            for c in contact_with_pad:
                penetration_depth = c[8]
                if penetration_depth < 0:
                    # The drone is penetrating the pad
                    print("Drone is penetrating the pad!")
                    reward -= 10  # penalty
                    self.crashed = True
                else:
                    # The drone is just in stable contact with the pad
                    print("Drone made stable contact (landed) on the pad!")
                    reward += self.leg_contact_reward
                    self.landed = True
           

        contact_with_ground = p.getContactPoints(self.drone, self.plane)
        if contact_with_ground:
            # This means the drone definitely touched the plane.
            # consider *any* contact with the plane as a crash, do:
            reward -= 5
            self.crashed = True

        

        return reward
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(action)
        p.applyExternalForce(self.drone, -1, action, [0, 0, 0], p.WORLD_FRAME)

        drone_position, drone_orientation = p.getBasePositionAndOrientation(self.drone)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-45,cameraTargetPosition=drone_position)

        p.stepSimulation()

        if args.visual_mode.upper() == "GUI":
            time.sleep(1./240.)
        # time.sleep(1./240.)
        

        observation = self.get_observation()
        reward = self.compute_reward(observation)
        terminated = self.is_done(observation) or getattr(self, "landed", False) or getattr(self, "crashed", False)
        truncated = False 

        return observation, reward, terminated, truncated, {}

    def is_done(self, observation):
        if self.landed or self.crashed:
            return True

        contact_with_ground = p.getContactPoints(self.drone, self.plane)

        if contact_with_ground:
            return True 

        if observation[2] <= 0:
            return True

        if abs(observation[0]) > self.boundary_limits or abs(observation[1]) > self.boundary_limits or observation[2] > self.boundary_limits:
            return True

        return False
    
    def close(self):
        p.disconnect()


def main():
    env = Monitor(DroneLandingEnv())
    tensorboard_log_dir = args.tensorboard_log_dir
    # find the algorithm to benchmark
    algorithm_name = args.algorithm_name 
    if algorithm_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
    elif algorithm_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
    # elif algorithm_name == "TRPO":
    #     model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
    else:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}, Please give right algorithm name" )
    
    eval_callback = EvalCallback(env, best_model_save_path='./logs_metrics_benchmark_tensorboard/',
                                 log_path='./logs_metrics_benchmark_tensorboard/', eval_freq=10000, verbose=1)
    
    # train the model
    model.learn(total_timesteps=2e6, callback = eval_callback, progress_bar = True)
    model.save(f"{args.model_name_to_save}")
    
    env.close()

    if algorithm_name == "PPO":
        # reload the model and test it 
        model = PPO.load(f"{args.model_name_to_save}")
        # for testing use GUI mode
        args.visual_mode = "GUI"
        env = DroneLandingEnv()

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    elif algorithm_name == "A2C":
        # reload the model and test it 
        model = A2C.load(f"{args.model_name_to_save}")
        # for testing use GUI mode
        args.visual_mode = "GUI"
        env = DroneLandingEnv()

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    else:
        pass 
        

    env.close()


if __name__ == "__main__":
    main()