import argparse
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from sb3_contrib import ARS, CrossQ, TQC, TRPO
# from Drone_Controller_PID import DroneControllerPID  
from Drone_Controller_PID_Wind import DroneControllerPID  

# Initialize environment
env = DroneControllerPID()
# model_path = "drone_landing_model_using_ppo"
model_path = "./logs_metrics_benchmark_tensorboard/best_model"
model = PPO.load(model_path, env=env, device='cpu')
# model = A2C.load(model_path, env=env, device='cpu')
# model = DDPG.load(model_path, env=env, device='cpu')
# model = TD3.load(model_path, env=env, device='cpu')
# model = SAC.load(model_path, env=env, device='cpu')
# model = ARS.load(model_path, env=env, device='cpu')
# model = CrossQ.load(model_path, env=env, device='cpu')
# model = TQC.load(model_path, env=env, device='cpu')
# model = TRPO.load(model_path, env=env, device='cpu')


n_episodes = 10

# Evaluate episodes
for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(1/240) 
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()
