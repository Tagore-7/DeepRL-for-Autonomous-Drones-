import Drone_Landing_RL
from Drone_Landing_RL import DroneLandingEnv


# TD3 train and test
from stable_baselines3 import TD3

env = DroneLandingEnv()
tensorboard_log_dir = "./tensorboard_logs_all_metrics/"
model = TD3("MlpPolicy", env, verbose=1, device="cuda:1", tensorboard_log=tensorboard_log_dir)
model.learn(total_timesteps=1000000)

# save the model
model.save("drone_landing_model_td3_all_metrics_action_param_3")
env.close()

# reload the model and test it
model = TD3.load("drone_landing_model_td3_all_metrics_action_param_3")
env = DroneLandingEnv()

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

env.close()
