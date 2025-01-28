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
from Base_Drone_Controller import BaseDroneController

from DSLPIDControl import DSLPIDControl
from enums import DroneModel

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
                        help = 'Discount factor (gamma) for the RL algorithm',
    )
    return parser.parse_args()

args = parse_args()

class DroneControllerVelocity(BaseDroneController):
    def __init__(self):
        super(DroneControllerVelocity, self).__init__(args=args)
        self.pid = DSLPIDControl(DroneModel.CF2X)

    def _actionSpace(self):
        # ---- [desired_vx, desired_vy, desired_vz, desired_yaw_rate] ---- #
        act_lower_bound = np.array([-1, -1, -1, -1], dtype=np.float32)
        act_upper_bound = np.array([ 1,  1,  1, 1], dtype=np.float32)
        self.action_space = Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        return self.action_space
    
    def _observationSpace(self):
        # ---- [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] ---- #
        obs_dim = 12
        high = np.array([np.finfo(np.float32).max] * obs_dim, dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        return self.observation_space

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_counter += 1
        
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone)

        # ---- Calculate distance to launch pad ---- #
        distance_to_pad_xy = np.linalg.norm(self.target_pos - np.array(position))
        # force_limit = self.MASS * -self.gravity * self.THRUST2WEIGHT_RATIO
        # v_max = math.sqrt(force_limit / self.KF)
        # speed_scale = min(distance_to_pad_xy, v_max)

        # Drone drops in Z axis immediately
        # Before attempting to fly in XY to launch pad.
        # Testing safe altitude here
        if distance_to_pad_xy > 2.0:
            safe_altitude = 2.0
            test_target_pos = max(self.target_pos[2], safe_altitude)
        else:
            test_target_pos = self.target_pos

        drone_position = np.array(position)
        drone_linear_vel = np.array(linear_vel)
        drone_angular_vel = np.array(angular_vel)
        drone_orientation = np.array(orientation)

        # ---- Calculate control actions based on position ---- #
        rpm, pos_error, yaw_error = self.pid.computeControl(
            control_timestep = self.time_step,
            cur_pos = drone_position,
            cur_quat = drone_orientation,
            cur_vel = drone_linear_vel,
            cur_ang_vel = drone_angular_vel,
            target_pos = test_target_pos,
        )
        #print(f"RPM: {rpm}, Position Error: {pos_error}, Yaw Error: {yaw_error}")

        # ---- Convert RPM to force and torque ---- #
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        #print(f"Forces: {forces}, Torques: {torques}")

        # ---- Calculating net yaw torque around z axis ---- #
        # ---- My understanding is that in a standard quadcopter layout, the front
        # ---- two rotors spin clockwise, rear spin counterclockwise.
        # ---- Net torque around z-axis depends on the sign of each rotor's torque?
        # ---- So summing up with alternating signs, you get the total yaw torque around the vertical axis?
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        # ---- Applying forces to each rotor ---- #
        for i in range(4):
            p.applyExternalForce(
                self.drone, 
                linkIndex=i, 
                forceObj=[0, 0, forces[i]],
                # posObj=[0, 0, 0], # apply force at the center of mass
                posObj=self.rotor_positions_local[i], # apply force at each rotor?
                flags=p.LINK_FRAME
            )
        
        # ---- Apply net torque about the z-axis ---- #
        p.applyExternalTorque(
            self.drone,
            -1,  # Using 4 for the center of mass link in URDF
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME
        )

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-45,cameraTargetPosition=position)
        p.stepSimulation()
        if args.visual_mode.upper() == "GUI":
            time.sleep(self.time_step)

        observation = self._get_observation()
        reward = self._compute_reward(observation, action)
        terminated = (
            self._is_done(observation) or self.step_counter >= self.max_steps or getattr(self, "crashed", False)
        )
        truncated = False

        return observation, reward, terminated, truncated, {}

def main():
    env = Monitor(DroneControllerVelocity())
    tensorboard_log_dir = args.tensorboard_log_dir
    gamma_value = args.discount_factor
    algorithm_name = args.algorithm_name 
    if algorithm_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, gamma = gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, gamma = gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    else:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}")

    eval_callback = EvalCallback(
        env,
        best_model_save_path=tensorboard_log_dir,
        log_path=tensorboard_log_dir,
        eval_freq=10000,
        verbose=1
    )

    # Train the model
    model.learn(total_timesteps=500000, callback=eval_callback, progress_bar=True)
    model.save(f"{args.model_name_to_save}")
    
    env.close()

if __name__ == "__main__":
    main()
