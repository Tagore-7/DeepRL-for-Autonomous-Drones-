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
    parser.add_argument(
        '--discount_factor',
        type = float,
        default = 0.99,
        help = 'Discount factor (gamma) for the RL algorithm',
    )
    

    return parser.parse_args()

args = parse_args()

class DroneLandingEnv(gym.Env):
    def __init__(self):
        super(DroneLandingEnv, self).__init__()

        if args.visual_mode.upper() == "GUI":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = None
        self.launch_pad = None
        self.launch_pad_position = args.launch_pad_position
        self.boundary_limits = args.boundary_limits
        self.distance_reward_weight = args.distance_reward_weight
        self.leg_contact_reward = args.leg_contact_reward
        self.landed = False
        self.crashed = False
        self.drone = None
        self.alpha = 0.1  # Hyperparameter for az
        self.c = 0  # Hyperparameter indicating landing state bonus
        self.step_counter = 0  # Step counter for termination condition
        self.max_steps = 10000  # Maximum steps per episode

        # Define action space (ax, ay)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Define observation space (position, velocity)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, args.gravity)
        self.plane = p.loadURDF("plane.urdf")
        self.launch_pad = p.loadURDF(
            "launch_pad.urdf", self.launch_pad_position, globalScaling=1.0, useFixedBase=True
        )
        self.drone = self.load_drone()
        self.landed = False
        self.crashed = False
        self.c = 0  # Reset the bonus parameter
        self.step_counter = 0

        return self.get_observation(), {}

    def load_drone(self):
        start_x = random.uniform(-15, 15)
        start_y = random.uniform(-15, 15)
        start_z = random.uniform(5, 15) 
        drone = p.loadURDF("cf2x.urdf", [start_x, start_y, start_z])
        return drone

    def get_observation(self):
        drone_position, _ = p.getBasePositionAndOrientation(self.drone)
        drone_linear_velocity, _ = p.getBaseVelocity(self.drone)

        return np.array([
            *drone_position,
            *drone_linear_velocity,
        ])

    def compute_reward(self, observation, action):
        px, py, pz = observation[0:3]  # Position
        vx, vy, vz = observation[3:6]  # Linear velocity
        ax, ay = action[0], action[1]  # Actions from the agent

        # Compute shaping reward
        shaping = (
            -100 * np.sqrt(px**2 + py**2 + pz**2)  # Distance penalty
            - 10 * np.sqrt(vx**2 + vy**2 + vz**2)  # Velocity penalty
        )

        # Check if drone has landed safely
        contact_points = p.getContactPoints(self.drone, self.launch_pad)
        if contact_points and abs(vz)  < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1:
            self.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay))  # Bonus for throttle tending to zero
            shaping +=  self.c
            self.landed = True

        # Reward difference (temporal difference shaping)
        if hasattr(self, 'previous_shaping'):
            reward = shaping - self.previous_shaping
        else:
            reward = shaping

        self.previous_shaping = shaping
        return reward

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_counter += 1

        # Desired velocities (ax, ay)
        ax = action[0]
        ay = action[1]

        # Compute az based on altitude
        drone_position, _ = p.getBasePositionAndOrientation(self.drone)
        az = -self.alpha * drone_position[2]  # Vertical velocity (smooth descent)

        roll, pitch = np.arcsin(ax), np.arcsin(ay)

        p.resetBasePositionAndOrientation(
            self.drone,
            drone_position,
            p.getQuaternionFromEuler([roll, pitch, 0])
        )


         # Simulate forward motion based on ax, ay
        velocity_x = ax
        velocity_y = ay
        velocity_z = az

        # Apply the velocities
        p.resetBaseVelocity(
            self.drone,
            linearVelocity=[velocity_x, velocity_y, velocity_z]
        )

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-45,cameraTargetPosition=drone_position)
        p.stepSimulation()
        if args.visual_mode.upper() == "GUI":
            time.sleep(1.0 / 240.0)

        observation = self.get_observation()
        reward = self.compute_reward(observation, action)
        terminated = (
            self.is_done(observation) or self.step_counter >= self.max_steps or getattr(self, "crashed", False)
        )
        truncated = False

        return observation, reward, terminated, truncated, {}

    def is_done(self, observation):
        px, py, pz = observation[0:3]

        if self.landed or self.crashed:
            return True

        contact_with_ground = p.getContactPoints(self.drone, self.plane)
        if contact_with_ground:
            self.crashed = True 
            return True

        if pz <= 0 or abs(px) > self.boundary_limits or abs(py) > self.boundary_limits or pz > self.boundary_limits:
            self.crashed = True 
            return True

        return False

    def close(self):
        p.disconnect()



def main():
    env = Monitor(DroneLandingEnv())
    tensorboard_log_dir = args.tensorboard_log_dir
    gamma_value = args.discount_factor
    # find the algorithm to benchmark
    algorithm_name = args.algorithm_name 
    if algorithm_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, gamma = gamma_value, tensorboard_log=tensorboard_log_dir, device = "cpu")
    elif algorithm_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, gamma = gamma_value, tensorboard_log=tensorboard_log_dir)
    # elif algorithm_name == "TRPO":
    #     model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
    else:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}, Please give right algorithm name" )
    
    eval_callback = EvalCallback(env, best_model_save_path='./logs_metrics_benchmark_tensorboard/',
                                 log_path='./logs_metrics_benchmark_tensorboard/', eval_freq=10000, verbose=1)
    
    # train the model
    model.learn(total_timesteps=1e6, callback = eval_callback, progress_bar = True)
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