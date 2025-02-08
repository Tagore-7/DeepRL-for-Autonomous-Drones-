import argparse
import os 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import xml.etree.ElementTree as ET
import math

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from Base_Drone_Controller import BaseDroneController

import multiprocessing
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


from DSLPIDControl import DSLPIDControl
from enums import DroneModel

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking DeepRL algorithms for drone landing task')
    parser.add_argument('--algorithm_name', type=str, default='PPO',
                        help='The name of the algorithm to benchmark. Options: PPO, A2C, DDPG, TD3, SAC, ARS, CROSSQ, TQC, TRPO')
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
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor (gamma) for the RL algorithm')
    return parser.parse_args()

args = parse_args()

class DroneControllerPID(BaseDroneController):
    def __init__(self):
        super(DroneControllerPID, self).__init__(args=args)
        self.pid = DSLPIDControl(DroneModel.CF2X)

    def _actionSpace(self):  # ax, ay, az
        act_lower_bound = np.array([-1, -1, -1], dtype=np.float32)
        act_upper_bound = np.array([ 1,  1,  1], dtype=np.float32)
        self.action_space = Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        return self.action_space
    
    def _observationSpace(self):
        # ---- [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] ---- #
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo])
        obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
        self.observation_space = Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        return self.observation_space

    def _preprocessAction(self, action):
        target = action
        
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        # rpy = p.getEulerFromQuaternion(orientation)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone)

        next_pos = self._calculateNextStep(
            current_position=position,
            destination=target,
            step_size=1
        )

        rpm, _, _ = self.pid.computeControl(
            control_timestep=self.time_step,
            cur_pos=position,
            cur_quat=orientation,
            cur_vel=linear_vel,
            cur_ang_vel=angular_vel,
            target_pos=next_pos
        )

        return rpm

    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        position, orientation = p.getBasePositionAndOrientation(self.drone)
        rpm = np.reshape(self._preprocessAction(action), 4)

        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM

        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(
                self.drone,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=self.rotor_positions_local[i],
                flags=p.LINK_FRAME,
            )
        p.applyExternalTorque(
            self.drone,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
        )

        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=position)
        p.stepSimulation()
        if args.visual_mode.upper() == "GUI":
            time.sleep(self.time_step)

        observation = self._get_observation()
        reward = self._compute_reward(observation, action)
        terminated = (
            self._is_done(observation) or self.step_counter >= self.max_steps or getattr(self, "crashed", False)
        )
        truncated = False

        self.step_counter += 1
        return observation, reward, terminated, truncated, {}

def main():
    num_cpu = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cpu}")
    env = make_vec_env(
        lambda: Monitor(DroneControllerPID()),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
        seed=42
    )

    tensorboard_log_dir = args.tensorboard_log_dir
    gamma_value = args.discount_factor
    algorithm_name = args.algorithm_name.upper()

    # Choose the model based on the algorithm name
    if algorithm_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "DDPG":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", env, verbose=1, gamma=gamma_value, action_noise=action_noise, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, verbose=1, gamma=gamma_value, action_noise=action_noise, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu", ent_coef='auto', target_entropy='auto')
    elif algorithm_name == "ARS":
        model = ARS("MlpPolicy", env, verbose=1, learning_rate=0.02, delta_std=0.05, n_delta=8, n_top=8, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "CROSSQ":
        model = CrossQ("MlpPolicy", env, verbose=1, gamma=gamma_value,
                       tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "TQC":
        policy_kwargs = dict(n_quantiles=25, n_critics=2)
        model = TQC("MlpPolicy", env, verbose=1, gamma=gamma_value,
                    tensorboard_log=tensorboard_log_dir, device="cpu",
                    top_quantiles_to_drop_per_net=2, policy_kwargs=policy_kwargs)
    elif algorithm_name == "TRPO":
        model = TRPO("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")

    else:
        raise ValueError(f"Invalid algorithm name: {args.algorithm_name}")

    # For ARS, asynchronous evaluation is experimental and callbacks are not fully supported.
    if algorithm_name not in ["ARS"]:
        eval_callback = EvalCallback(
            env,
            best_model_save_path=tensorboard_log_dir,
            log_path=tensorboard_log_dir,
            eval_freq=10000,
            verbose=1
        )
        model.learn(total_timesteps=2e6, callback=eval_callback, progress_bar=True)
    else:
        model.learn(total_timesteps=2e6, progress_bar=True)
    
    model.save(f"{args.model_name_to_save}")
    
    env.close()

if __name__ == "__main__":
    main()
