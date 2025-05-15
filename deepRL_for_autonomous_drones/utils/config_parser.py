import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking DeepRL algorithms for drone landing task")
    # parser.add_argument(
    #     "--algorithm_name",
    #     type=str,
    #     default="PPO",
    #     help="The name of the algorithm to benchmark. Options: PPO, A2C, DDPG, TD3, SAC, ARS, CROSSQ, TQC, TRPO",
    # )
    parser.add_argument(
        "--launch_pad_position",
        type=lambda x: np.array([float(i) for i in x.split(",")]),
        default=np.array([0, 0, 0]),
        help="The position of the launch pad for the drone to land on in the environment",
    )
    parser.add_argument(
        "--distance_reward_weight",
        type=float,
        default=2.0,
        help="The weight for the distance reward (distance between drone and launch pad)",
    )
    parser.add_argument(
        "--tensorboard_log_dir", type=str, default="./logs_metrics_benchmark_tensorboard/", help="The directory to store TensorBoard logs"
    )
    parser.add_argument("--boundary_limits", type=int, default=20, help="The boundary limits for the drone to fly in the environment")
    parser.add_argument("--gravity", type=float, default=-9.8, help="The gravity value for the environment")
    parser.add_argument("--leg_contact_reward", type=int, default=100, help="The reward for the drone making contact with the launch pad")
    parser.add_argument("--model_name_to_save", type=str, default="drone_landing_model_using_ppo", help="Name of the model to save")
    parser.add_argument("--visual_mode", type=str, default="DIRECT", help="Visual mode of the environment: GUI or DIRECT")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor (gamma) for the RL algorithm")
    parser.add_argument("--reward_function", type=int, default=1, help="Which reward function you want to use: 1 or 2 or 3")
    parser.add_argument("--debug_axes", type=bool, default=False, help="Draws visual lines for drone axes for debugging")
    parser.add_argument("--add_obstacles", type=bool, default=False, help="Determines if there will obstacles in the environment.")
    parser.add_argument("--enable_wind", type=bool, default=False, help="Determines if there will be wind effects applied to the drone.")
    parser.add_argument("--enable_curriculum_learning", type=bool, default=False, help="Determines if curriculum learning will be used.")
    parser.add_argument(
        "--observation_type",
        type=int,
        default=1,
        help="Which observation type to use. 1: Kinematic, 2: Kin+LiDAR, 3: Kin+RGB, 4: Kin+LiDAR+RGB",
    )

    return parser.parse_args()
