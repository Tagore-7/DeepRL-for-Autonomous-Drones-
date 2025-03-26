import argparse
import multiprocessing
import numpy as np

# import pkg_resources
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.custom_feature_extractor import (
    CustomFeatureExtractor,
)
from deepRL_for_autonomous_drones.utils.Custom_Callbacks import (
    EpisodeRewardCallback,
    ToggleWindCallback,
    ToggleTreesCallback,
    # ToggleStaticBlocksCallback,
    # ToggleDonutObstaclesCallback,
    # ToggleMovingBlocksCallback,
    SaveModelCallback,
)


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


def main():
    num_cpu = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cpu}")
    if args.visual_mode.upper() == "GUI":
        num_cpu = 1
        env = Monitor(DroneControllerRPM(args=args))
    else:
        env = make_vec_env(
            lambda: Monitor(DroneControllerRPM(args=args)),
            n_envs=num_cpu,
            vec_env_cls=SubprocVecEnv,
            seed=42,
        )

    tensorboard_log_dir = args.tensorboard_log_dir
    gamma_value = args.discount_factor
    algorithm_name = args.algorithm_name.upper()

    # ---- Custom callbacks ----#
    reward_callback = EpisodeRewardCallback()
    eval_callback = EvalCallback(
        env,
        best_model_save_path=tensorboard_log_dir,
        log_path=tensorboard_log_dir,
        eval_freq=10000,
        verbose=1,
    )

    if args.add_obstacles and args.enable_wind:
        toggle_trees = (
            ToggleTreesCallback(threshold=int(5e6))
            if args.enable_curriculum_learning
            else ToggleTreesCallback(threshold=int(0))
        )
        toggle_wind = (
            ToggleWindCallback(threshold=int(15e6))
            if args.enable_curriculum_learning
            else ToggleWindCallback(threshold=int(0))
        )
        # ---- Save models at specific timesteps ----#
        save_thresholds = [5e6, 10e6, 15e6]
        save_paths = [
            "landing_model_5M",
            "landing_model_trees_10M",
            "landing_model_trees_wind_15M",
        ]
        save_callback = SaveModelCallback(thresholds=save_thresholds, save_paths=save_paths)

        total_timesteps = 25e6
        callback = [
            reward_callback,
            toggle_wind,
            toggle_trees,
        ]
        if args.enable_curriculum_learning:
            callback.append(save_callback)
    elif args.add_obstacles:
        toggle_trees = (
            ToggleTreesCallback(threshold=int(5e6))
            if args.enable_curriculum_learning
            else ToggleTreesCallback(threshold=int(0))
        )
        save_thresholds = [5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 11e6, 12e6, 13e6, 14e6]
        save_paths = [
            "landing_model_5M",
            "landing_model_6M",
            "landing_model_7M",
            "landing_model_8M",
            "landing_model_9M",
            "landing_model_10M",
            "landing_model_11M",
            "landing_model_12M",
            "landing_model_13M",
            "landing_model_14M",
        ]
        save_callback = SaveModelCallback(thresholds=save_thresholds, save_paths=save_paths)

        total_timesteps = 10e6
        callback = [reward_callback, toggle_trees]
        if args.enable_curriculum_learning:
            callback.append(save_callback)
    elif args.enable_wind:
        toggle_wind = (
            ToggleWindCallback(threshold=int(5e6))
            if args.enable_curriculum_learning
            else ToggleWindCallback(threshold=int(0))
        )
        # ---- Save models at specific timesteps ----#
        save_thresholds = [5e6, 10e6]
        save_paths = [
            "landing_model_5M",
            "landing_model_wind_10M",
        ]
        save_callback = SaveModelCallback(thresholds=save_thresholds, save_paths=save_paths)

        total_timesteps = 10e6
        callback = [reward_callback, toggle_wind]
        if args.enable_curriculum_learning:
            callback.append(save_callback)
    else:
        total_timesteps = 5e6
        callback = [reward_callback]

    # ---- Choose the model based on the algorithm name ----#
    if algorithm_name == "PPO":
        if args.observation_type != 1 or args.observation_type != 2:
            model = PPO(
                "MultiInputPolicy",
                env,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=gamma_value,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                device="auto",
                policy_kwargs=dict(
                    features_extractor_class=CustomFeatureExtractor,
                    net_arch=[
                        512,
                        256,
                    ],  # For PPO, both actor and critic share pi and vf layers, no need to specify different
                ),
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=gamma_value,
                # use_amp = True,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                device="cpu",
            )
    elif algorithm_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    elif algorithm_name == "DDPG":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            action_noise=action_noise,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    elif algorithm_name == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            action_noise=action_noise,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    elif algorithm_name == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
            ent_coef="auto",
            target_entropy="auto",
        )
    elif algorithm_name == "ARS":
        model = ARS(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.02,
            delta_std=0.05,
            n_delta=8,
            n_top=8,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    elif algorithm_name == "CROSSQ":
        model = CrossQ(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    elif algorithm_name == "TQC":
        policy_kwargs = dict(n_quantiles=25, n_critics=2)
        model = TQC(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
            top_quantiles_to_drop_per_net=2,
            policy_kwargs=policy_kwargs,
        )
    elif algorithm_name == "TRPO":
        model = TRPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=gamma_value,
            tensorboard_log=tensorboard_log_dir,
            device="cpu",
        )
    else:
        raise ValueError(f"Invalid algorithm name: {args.algorithm_name}")

    # For ARS, asynchronous evaluation is experimental and callbacks are not fully supported.
    if algorithm_name not in ["ARS"]:
        callback.append(eval_callback)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    else:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    model.save(f"{args.model_name_to_save}")

    env.close()


if __name__ == "__main__":
    main()
