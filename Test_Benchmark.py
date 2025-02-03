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
from Drone_Controller_Velocity import DroneControllerVelocity
from Drone_Controller_PID import DroneControllerPID

import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking DeepRL algorithms for drone landing task')
    parser.add_argument('--algorithm_name', type=str, default='PPO',
                        help='The name of the algorithm to benchmark. Options: PPO, A2C')
    parser.add_argument('--boundary_limits', type=int, default=20,
                        help='The boundary limits for the drone to fly in the environment')
    parser.add_argument('--launch_pad_position', type=lambda x: np.array([float(i) for i in x.split(",")]),
                        default=np.array([0, 0, 0]),
                        help='The position of the launch pad for the drone to land on in the environment')
    parser.add_argument('--gravity', type=float, default=-9.8,
                        help='The gravity value for the environment')
    parser.add_argument('--distance_reward_weight', type=float, default=2.0,
                        help='The weight for the distance reward (distance between drone and launch pad)')
    parser.add_argument('--leg_contact_reward', type=int, default=100,
                        help='The reward for the drone making contact with the launch pad')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./logs_metrics_benchmark_tensorboard/',
                        help='The directory to store TensorBoard logs')
    parser.add_argument('--model_name_to_save', type=str, default='drone_landing_model_using_ppo',
                        help='Name of the model to save')
    parser.add_argument('--visual_mode', type=str, default="DIRECT",
                        help='Visual mode of the environment: GUI or DIRECT')
    parser.add_argument('--discount_factor', type = float, default = 0.99,
                        help = 'Discount factor (gamma) for the RL algorithm',)

    return parser.parse_args()

args = parse_args()

def step(self, action):
  if args.visual_mode.upper() == "GUI":
    time.sleep(1.0 / 240.0)

def main():
    global args
    args = parse_args()
    algorithm_name = args.algorithm_name 

    if algorithm_name == "PPO":
        # reload the model and test it 
        model = PPO.load(f"{args.model_name_to_save}")
        # model = PPO.load("./logs_metrics_benchmark_tensorboard/best_model")
        # for testing use GUI mode
        args.visual_mode = "GUI"
        env = DroneControllerPID()

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    elif algorithm_name == "A2C":
        # reload the model and test it 
        model = A2C.load(f"{args.model_name_to_save}")
        # for testing use GUI mode
        args.visual_mode = "GUI"
        env = DroneControllerPID()

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    else:
        pass 

    env.close()

if __name__ == "__main__":
    main()
