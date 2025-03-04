import argparse
import time
import numpy as np
import pybullet as p
import pkg_resources
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM

# Initialize environment
env = DroneControllerRPM()

model = PPO.load(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'envs/logs_metrics_benchmark_tensorboard/best_model'), env=env, device='cpu')
# model = PPO.load(pkg_resources.resource_filename('deepRL_for_autonomous_drones', 'envs/drone_landing_model_using_ppo'), env=env, device='cpu')

# model = A2C.load(model_path, env=env, device='cpu')
# model = DDPG.load(model_path, env=env, device='cpu')
# model = TD3.load(model_path, env=env, device='cpu')
# model = SAC.load(model_path, env=env, device='cpu')
# model = ARS.load(model_path, env=env, device='cpu')
# model = CrossQ.load(model_path, env=env, device='cpu')
# model = TQC.load(model_path, env=env, device='cpu')
# model = TRPO.load(model_path, env=env, device='cpu')


n_episodes = 100

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
