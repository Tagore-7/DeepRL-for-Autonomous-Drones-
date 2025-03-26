import argparse
import time
import numpy as np
import pkg_resources
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking DeepRL algorithms for drone landing task"
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="PPO",
        help="The name of the algorithm to benchmark. Options: PPO, A2C, DDPG, TD3, SAC, ARS, CROSSQ, TQC, TRPO",
    )
    parser.add_argument(
        "--boundary_limits",
        type=int,
        default=20,
        help="The boundary limits for the drone to fly in the environment",
    )
    parser.add_argument(
        "--launch_pad_position",
        type=lambda x: np.array([float(i) for i in x.split(",")]),
        default=np.array([0, 0, 0]),
        help="The position of the launch pad for the drone to land on in the environment",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=-9.8,
        help="The gravity value for the environment",
    )
    parser.add_argument(
        "--distance_reward_weight",
        type=float,
        default=2.0,
        help="The weight for the distance reward (distance between drone and launch pad)",
    )
    parser.add_argument(
        "--leg_contact_reward",
        type=int,
        default=100,
        help="The reward for the drone making contact with the launch pad",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="./logs_metrics_benchmark_tensorboard/",
        help="The directory to store TensorBoard logs",
    )
    parser.add_argument(
        "--model_name_to_save",
        type=str,
        default="drone_landing_model_using_ppo",
        help="Name of the model to save",
    )
    parser.add_argument(
        "--visual_mode",
        type=str,
        default="DIRECT",
        help="Visual mode of the environment: GUI or DIRECT",
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.99,
        help="Discount factor (gamma) for the RL algorithm",
    )
    parser.add_argument(
        "--reward_function",
        type=int,
        default=1,
        help="Which reward function you want to use: 1 or 2 or 3",
    )
    parser.add_argument(
        "--debug_axes",
        type=bool,
        default=False,
        help="Draws visual lines for drone axes for debugging",
    )
    parser.add_argument(
        "--add_obstacles",
        type=bool,
        default=False,
        help="Determines if there will obstacles",
    )
    parser.add_argument(
        "--enable_wind",
        type=bool,
        default=False,
        help="Determines if there will be wind effects applied to the drone",
    )
    parser.add_argument(
        "--enable_ground_effect",
        type=bool,
        default=False,
        help="Determines if there will be ground effects applied to the drone",
    )
    parser.add_argument(
        "--enable_curriculum_learning",
        type=bool,
        default=False,
        help="Determines if curriculum learning will be used. If it is, obstacles and wind activation will be delayed",
    )
    parser.add_argument(
        "--observation_type",
        type=int,
        default=1,
        help="Which observation type to use. 1: Kinematic, 2: Kin+LiDAR, 3: Kin+RGB, 4: Kin+LiDAR+RGB",
    )
    return parser.parse_args()


args = parse_args()

# Initialize environment
env = DroneControllerRPM(args)

# env.setWindEffects(True)
# env.setStaticBlocks(True)
# env.setDonutObstacles(True)
# env.setMovingBlocks(True)

model = PPO.load(
    pkg_resources.resource_filename(
        "deepRL_for_autonomous_drones",
        "envs/logs_metrics_benchmark_tensorboard/PPO_12/drone_landing_model_using_ppo",
    ),
    env=env,
    device="cpu",
)
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
        time.sleep(1 / 240)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()
