import numpy as np
import pandas as pd 
import torch 
import os 
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import seaborn as sns 
import matplotlib.pyplot as plt 
from deepRL_for_autonomous_drones import envs

def make_env(task: str, wind_level: str = "none", render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)
    env = FlattenObservation(env)
    # if hasattr(env.unwrapped, "setWindEffects"):
    #     env.unwrapped.setWindEffects(True)
    # if hasattr(env.unwrapped, "setWindLevel"):
    #     env.unwrapped.setWindLevel(wind_level)
    return env

def make_env_with_wind(task: str, wind_level: str = "none", render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)
    env = FlattenObservation(env)
    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(True)
    if hasattr(env.unwrapped, "setWindLevel"):
        env.unwrapped.setWindLevel(wind_level)
    return env


class ObservationDisruptor:
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def applyGussianDisruptor(self, obs: np.ndarray):
        noise = np.random.normal(0.0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return obs, noise, noisy_obs
    
    def dropoutDisruptor(self, obs: np.ndarray, dropout_prob: float = 0.3, min_scale: float = 0.1, max_scale: float = 0.3):
        mask = np.random.binomial(1, 1 - dropout_prob, size = obs.shape)
        scale_factors = np.random.uniform(min_scale, max_scale, size = obs.shape)
        dropped_obs = obs * (mask + ( 1- mask) * scale_factors)
        return obs, mask, dropped_obs
    


    
def evaluate_agent_with_bias_drift(
    model_path: str,
    env_name: str,
    render: bool = False,
    episodes: int = 20,
    drift_magnitude: float = 0.2,
    drift_type: str = "constant",  # or "random_walk"
    save_path: str = "eval_results_with_bias_drift.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)

    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False
        done = False
        total_reward, total_cost, steps = 0, 0, 0

        if drift_type == "constant":
            bias = np.ones_like(obs) * drift_magnitude
        elif drift_type == "random_walk":
            bias = np.random.normal(0.0, drift_magnitude, size=obs.shape)
        else:
            raise ValueError("drift_type must be 'constant' or 'random_walk'")

        while not done:
            biased_obs = obs + bias
            action, _ = model.predict(biased_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cost = info.get("cost", None)
            total_reward += reward
            total_cost += cost
            steps += 1

            if drift_type == "random_walk":
                bias += np.random.normal(0.0, drift_magnitude * 0.01, size=bias.shape)

            logs.append({
                "episode": episode,
                "step": steps,
                "reward": reward,
                "cost": cost,
                "seed": episode,
                "original_obs": obs.tolist(),
                "bias": bias.tolist(),
                "biased_obs": biased_obs.tolist(),
                "obs_lidar": obs[12:76].tolist(),
                "biased_lidar": biased_obs[12:76].tolist(),
            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_reward,
            "cost": total_cost,
            "seed": episode,
            "original_obs": None,
            "bias": None,
            "biased_obs": None,
            "obs_lidar": None,
            "biased_lidar": None,
            "landed": env.unwrapped.landed,
        })

        df = pd.DataFrame(logs)
        df.to_csv(save_path, index=False)
        print(f"Saved bias drift evaluation to {save_path}")


def compare_with_bias_drift_results(
    normal_results_file_path: str,
    bias_drift_results_file_path: str,
    save_path: str = "eval_comparison_with_bias_drift.csv"
):
    normal_df = pd.read_csv(normal_results_file_path)
    bias_df = pd.read_csv(bias_drift_results_file_path)

    normal_final = normal_df[normal_df["step"] == "final"].copy()
    bias_final = bias_df[bias_df["step"] == "final"].copy()

    assert len(normal_final) == len(bias_final), "Mismatch in number of episodes"

    summary = []
    for i in range(len(normal_final)):
        summary.append({
            "episode": int(normal_final.iloc[i]["episode"]),
            "reward_clean": normal_final.iloc[i]["reward"],
            "cost_clean": normal_final.iloc[i]["cost"],
            "reward_bias": bias_final.iloc[i]["reward"],
            "cost_bias": bias_final.iloc[i]["cost"],
            "reward_diff": bias_final.iloc[i]["reward"] - normal_final.iloc[i]["reward"],
            "cost_diff": bias_final.iloc[i]["cost"] - normal_final.iloc[i]["cost"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"Saved bias drift comparison summary to {save_path}")


def visualize_performance_bias(results_file_path: str):
    df = pd.read_csv(results_file_path)
    df["episode"] = df["episode"].astype(int)

    # Stacked bar plot of reward
    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Bias Drift": df["reward_bias"]
    }, index=df["episode"])

    reward_stack.plot(kind="bar", stacked=True, colormap="Set2", figsize=(14, 6))
    plt.title("Stacked Episode Rewards: Clean + Bias Drift")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_bias_drift.png")

    # Reward difference bar plot
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y="reward_diff", palette="coolwarm")
    plt.title("Reward Difference (Bias Drift - Clean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_diff_bar_bias_drift.png")

    # Cost boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_bias"]])
    plt.title("Cost Distribution: Clean vs Bias Drift")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_boxplot_bias_drift.png")

    # Reward boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_bias"]])
    plt.title("Reward Distribution: Clean vs Bias Drift")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_boxplot_bias_drift.png")

def evaluate_agent_with_clipping(
        model_path: str,
        env_name: str,
        render: bool = False,
        episodes: int = 20,
        clip_min: float = -2,
        clip_max: float = 7,
        save_path: str = "eval_results_with_clipping.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)

    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False
        done = False
        total_reward, total_cost, steps = 0, 0, 0

        while not done:
            original_obs = obs.copy()
            clipped_obs = np.clip(original_obs, clip_min, clip_max)

            action, _ = model.predict(clipped_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cost = info.get("cost", None)
            total_reward += reward
            total_cost += cost
            steps += 1

            logs.append({
                "episode": episode,
                "step": steps,
                "reward": reward,
                "cost": cost,
                "seed": episode,
                "original_obs": original_obs.tolist(),
                "clipped_obs": clipped_obs.tolist(),
                "original_lidar": original_obs[12:76].tolist(),
                "clipped_lidar": clipped_obs[12:76].tolist()
            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_reward,
            "cost": total_cost,
            "seed": episode,
            "original_obs": None,
            "clipped_obs": None,
            "original_lidar": None,
            "clipped_lidar": None,
            "landed": env.unwrapped.landed
        })

        df = pd.DataFrame(logs)
        df.to_csv(save_path, index=False)
        print(f"Saved clipping evaluation to {save_path}")


def compare_with_clipping_results(
        normal_results_file_path: str,
        clipping_results_file_path: str,
        save_path: str = "eval_comparison_with_clipping.csv"
):
    normal_df = pd.read_csv(normal_results_file_path)
    clipping_df = pd.read_csv(clipping_results_file_path)

    normal_final = normal_df[normal_df["step"] == "final"].copy()
    clipping_final = clipping_df[clipping_df["step"] == "final"].copy()

    assert len(normal_final) == len(clipping_final), "Mismatch in number of episodes"

    summary = []
    for i in range(len(normal_final)):
        summary.append({
            "episode": int(normal_final.iloc[i]["episode"]),
            "reward_clean": normal_final.iloc[i]["reward"],
            "cost_clean": normal_final.iloc[i]["cost"],
            "reward_clipping": clipping_final.iloc[i]["reward"],
            "cost_clipping": clipping_final.iloc[i]["cost"],
            "reward_diff": clipping_final.iloc[i]["reward"] - normal_final.iloc[i]["reward"],
            "cost_diff": clipping_final.iloc[i]["cost"] - normal_final.iloc[i]["cost"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"Saved clipping comparison summary to {save_path}")


def visualize_performance_clipping(results_file_path: str):
    df = pd.read_csv(results_file_path)
    df["episode"] = df["episode"].astype(int)

    # Stacked bar plot of rewards
    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Clipping": df["reward_clipping"]
    }, index=df["episode"])
    reward_stack.plot(kind="bar", stacked=True, colormap="Paired", figsize=(14, 6), color=["skyblue", "darkorange"])
    plt.title("Stacked Episode Reward: Clipping + Clean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_clipping_comparison.png")

    # Reward difference bar plot
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y="reward_diff", palette="coolwarm")
    plt.title("Reward Difference (Clipping - Clean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/clipping_reward_difference_bar_plot.png")

    # Cost boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_clipping"]])
    plt.title("Cost Distribution: Clean vs Clipping")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/clipping_cost_boxplot.png")

    # Reward boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_clipping"]])
    plt.title("Reward Distribution: Clean vs Clipping")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/clipping_reward_boxplot.png")


def evaluate_agent_with_dropout_effect(
        model_path: str,
        env_name: str,
        render: bool = False,
        episodes: int = 20,
        dropout_prob: float = 0.5,
        min_scale: float = 0.1,
        max_scale: float = 0.3,
        save_path: str = "eval_results_with_dropout_effect.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)
    disruptor = ObservationDisruptor()

    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False 
        done = False 
        total_reward, total_cost, steps = 0, 0, 0 

        while not done:
            original_obs, mask, dropped_obs = disruptor.dropoutDisruptor(
                obs, 
                dropout_prob=dropout_prob,
                min_scale=min_scale,
                max_scale=max_scale
            )
            action, _ = model.predict(dropped_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cost = info.get("cost", None)
            total_reward += reward
            total_cost += cost 
            steps += 1 

            logs.append({
                "episode": episode,
                "step": steps,
                "reward": reward,
                "cost": cost,
                "seed": episode,
                "original_obs": original_obs.tolist(),
                "obs_lidar": original_obs[12: 76].tolist(),
                "mask": mask.tolist(),
                "dropped_obs": dropped_obs.tolist(),
                "dropped_lidar": dropped_obs[12:76].tolist(),
            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_reward,
            "cost": total_cost,
            "seed": episode,
            "original_obs": None,
            "obs_lidar": None,
            "mask": None,
            "dropped_obs": None,
            "dropped_lidar": None,
            "landed": env.unwrapped.landed
        })

        df = pd.DataFrame(logs)
        df.to_csv(save_path, index=False)
        print(f"Saved dropout evaluation to {save_path}")
    
    

def evaluate_agent_with_wind_effect(
        model_path: str,
        env_name: str, 
        render: bool = False,
        episodes: int = 20, 
        save_path: str = "eval_results_with_wind_effect.csv"
):
    model = PPO.load(model_path, device="cpu")
    wind_level_map = {
        0.0: "none",
        2.24: "light_breeze",
        4.47: "light_wind",
        8.94: "medium_wind",
        17.88: "high_wind",
    }
    results = []

    for wind_val, wind  in wind_level_map.items():
        env = make_env_with_wind(env_name, wind_level=str(wind), render=render)

        total_reward, total_cost, landings = 0, 0, 0 

        for episode in range(episodes):
            obs, _ = env.reset(seed = episode)
            env.unwrapped.landed = False 
            done = False 

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terimated, truncated, info = env.step(action)
                done = terimated or truncated
                total_reward += reward
                total_cost += info.get("cost", 0.0)

            if hasattr(env.unwrapped, "landed") and env.unwrapped.landed:
                landings += 1

        success_rate = landings / episodes * 100 

        results.append({
            "wind_level" : wind,
            "wind_value": wind_val,
            "total_reward": total_reward,
            "total_cost": total_cost, 
            "landing_success_rate": success_rate
        })

        env.close()

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Wind effect evaluation to {save_path}")

def evaluate_agent_with_gussian_noise(
        model_path: str,
        env_name: str, 
        render: bool = False,
        episodes: int = 20, 
        noise_std: float = 0.05,
        save_path: str = "eval_obs_gussian_noise_results.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)
    gussian_disruptor = ObservationDisruptor(noise_std=noise_std)

    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False 
        done = False
        total_noise_reward, total_noise_cost, steps = 0, 0, 0,

        while not done:
            original_obs, noise, noisy_obs = gussian_disruptor.applyGussianDisruptor(obs)
            noisy_action, _ =  model.predict(noisy_obs, deterministic = True)
            obs, reward, terminated, truncated, info = env.step(noisy_action)
            done = terminated or truncated
            cost = info.get("cost", None)
             
            total_noise_reward += reward 
            total_noise_cost += cost 

            steps +=1 

            logs.append({
                "episode" : episode,
                "step" : steps,
                "reward" : reward,
                "cost": cost,
                "seed" : episode,
                "obs_original_state" : original_obs.tolist(),
                "obs_original_lidar_state" : original_obs[12:76].tolist(),
                "noisy_of_state": noise.tolist(),
                "noisy_of_lidar": noise[12:76].tolist(),
                "gussian_obs_state" : noisy_obs.tolist(),
                "gussian_lidar_obs_state": noisy_obs[12:76].tolist(),

            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_noise_reward,
            "cost": total_noise_cost,
            "seed": episode,
            "obs_state": None,
            "obs_lidar": None,
            "noise_state": None,
            "noise_lidar": None,
            "noisy_obs_state": None,
            "noisy_obs_lidar": None,
            "landed": env.unwrapped.landed
        })


        
        df = pd.DataFrame(logs)
        df.to_csv(save_path, index=False)
        print(f"Saved evaluation to {save_path}") 

def evaluate_agent_without_noise(
    model_path: str,
    env_name: str, 
    render : bool = False,
    episodes: int = 20,
    save_path: str = "eval_obs_without_noise_results.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)

    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False
        done = False
        total_reward, total_cost, steps = 0, 0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = info.get("cost", None)

            total_reward += reward
            total_cost += cost
            steps += 1

            logs.append({
                "episode": episode,
                "step": steps,
                "reward": reward,
                "cost": cost,
                "seed": episode,
                "obs_state": obs.tolist(),
                "obs_lidar": obs[12:76].tolist(),
            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_reward,
            "cost": total_cost,
            "seed": episode,
            "obs_state": None,
            "obs_lidar": None,
            "landed": env.unwrapped.landed
        })

    df = pd.DataFrame(logs)
    df.to_csv(save_path, index=False)
    print(f"Saved original observation evaluation to {save_path}")

def evaluate_agent_with_observation_delay(
        model_path: str,
        env_name: str,
        render: bool = False,
        episodes: int = 20,
        delay_frames: int = 5,
        save_path: str = "eval_obs_delay_results.csv"
):
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, "none", render)
    
    logs = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        env.unwrapped.landed = False 
        delayed_obs = obs.copy()
        delay_counter = 0

        done = False 
        total_reward, total_cost, steps = 0, 0, 0

        while not done:
            action, _ = model.predict(delayed_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = info.get("cost", None)

            delay_counter +=1 
            if delay_counter >= delay_frames:
                delayed_obs = obs.copy()
                delay_counter = 0 

            total_reward += reward
            total_cost += cost 
            steps += 1 

            logs.append({
                "episode": episode,
                "step": steps,
                "reward": reward,
                "cost": cost,
                "seed": episode,
                "current_obs": obs.tolist(),
                "used_obs": delayed_obs.tolist(),
                "obs_lidar_actual": obs[12:76].tolist(),
                "obs_lidar_used": delayed_obs[12:76].tolist(),
            })

        logs.append({
            "episode": episode,
            "step": "final",
            "reward": total_reward,
            "cost": total_cost,
            "seed": episode,
            "current_obs": None,
            "used_obs": None,
            "obs_lidar_actual": None,
            "obs_lidar_used": None,
            "landed": env.unwrapped.landed
        })


        df = pd.DataFrame(logs)
        df.to_csv(save_path, index=False)
        print(f"saved delayed observation evaluation to {save_path}")


def compare_with_and_without_wind_results(
        normal_results_file_path: str,
        wind_effects_results_file_path: str,
        save_path: str = "eval_comparasion_btw_with_and_without_wind_effect.csv"
):
    normal_df = pd.read_csv(normal_results_file_path)
    wind_df = pd.read_csv(wind_effects_results_file_path)

    avg_clean_reward = normal_df[normal_df["step"] == "final"]["reward"].mean()
    avg_clean_cost = normal_df[normal_df["step"] == "final"]["cost"].mean()
    landing_clean = normal_df[normal_df["step"] == "final"]["landed"].mean() * 100 

    summary = []

    for _, row in wind_df.iterrows():
        summary.append({
            "wind_level": row["wind_level"],
            "reward_clean": avg_clean_reward,
            "cost_clean": avg_clean_cost,
            "landing_rate_clean": landing_clean,
            "reward_wind": row["total_reward"],
            "cost_wind": row["total_cost"],
            "landing_rate_wind": row["landing_success_rate"],
            "reward_diff": row["total_reward"] - avg_clean_reward,
            "cost_diff": row["total_cost"] - avg_clean_cost,
            "landing_rate_diff": row["landing_success_rate"] - landing_clean
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"Saved wind comparison summary to {save_path}")


def compare_with_and_without_noise_results(
        normal_results_file_path: str,
        noisy_results_file_path: str,
        save_path: str = "eval_comparasion_btw_with_and_without_gussian_noise.csv"
):
    normal_results_df = pd.read_csv(normal_results_file_path)
    noisy_results_df = pd.read_csv(noisy_results_file_path) 

    normal_results = normal_results_df[normal_results_df["step"] == "final"].copy()
    noisy_results = noisy_results_df[noisy_results_df["step"] == "final"].copy()

    assert len(normal_results) == len(noisy_results), "Mismatch in number of episdoes"

    summary = []

    for i in range(len(normal_results)):
        summary.append({
            "episode": int(normal_results.iloc[i]["episode"]),
            "reward_clean": normal_results.iloc[i]["reward"],
            "cost_clean": normal_results.iloc[i]["cost"],
            "reward_noisy": noisy_results.iloc[i]["reward"],
            "cost_noisy": noisy_results.iloc[i]["cost"],
            "reward_diff": noisy_results.iloc[i]["reward"] - normal_results.iloc[i]["reward"],
            "cost_diff": noisy_results.iloc[i]["cost"] - normal_results.iloc[i]["cost"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"Saved comparison summary to {save_path}")


def compare_with_delay_results(
        normal_results_file_path: str,
        delay_results_file_path:str,
        save_path: str = "eval_comparison_with_delay.csv"
):
    normal_results_df = pd.read_csv(normal_results_file_path)
    delay_results_df = pd.read_csv(delay_results_file_path)

    normal_final = normal_results_df[normal_results_df["step"] == "final"].copy()
    delay_final = delay_results_df[delay_results_df["step"] == "final"].copy()

    assert len(normal_final) == len(delay_final), "Mismatch in number of episodes"

    summary = []
    for i in range(len(normal_final)):
        summary.append({
            "episode": int(normal_final.iloc[i]["episode"]),
            "reward_clean": normal_final.iloc[i]["reward"],
            "cost_clean": normal_final.iloc[i]["cost"],
            "reward_delay": delay_final.iloc[i]["reward"],
            "cost_delay": delay_final.iloc[i]["cost"],
            "reward_diff": delay_final.iloc[i]["reward"] - normal_final.iloc[i]["reward"],
            "cost_diff": delay_final.iloc[i]["cost"] - normal_final.iloc[i]["cost"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"saved delay comparison summary to {save_path}")


def compare_with_dropout_results(
        normal_results_file_path: str,
        dropout_results_file_path: str,
        save_path: str = "eval_comparison_with_dropout.csv"
):
    normal_results_df = pd.read_csv(normal_results_file_path)
    dropout_results_df = pd.read_csv(dropout_results_file_path)

    normal_final = normal_results_df[normal_results_df["step"] == "final"].copy()
    dropout_final = dropout_results_df[dropout_results_df["step"] == "final"].copy()

    assert len(normal_final) == len(dropout_final), "Mismatch in number of episodes"

    summary = []
    for i in range(len(normal_final)):
        summary.append({
            "episode": int(normal_final.iloc[i]["episode"]),
            "reward_clean": normal_final.iloc[i]["reward"],
            "cost_clean": normal_final.iloc[i]["cost"],
            "reward_dropout": dropout_final.iloc[i]["reward"],
            "cost_dropout": dropout_final.iloc[i]["cost"],
            "reward_diff": dropout_final.iloc[i]["reward"] - normal_final.iloc[i]["reward"],
            "cost_diff": dropout_final.iloc[i]["cost"] - normal_final.iloc[i]["cost"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_path, index=False)
    print(f"saved dropout comparsion sumary to {save_path}") 


def visualize_performance_dropout(results_file_path: str):
    df = pd.read_csv(results_file_path)
    df["episode"] = df["episode"].astype(int)

    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Dropout": df["reward_dropout"]
    }, index=df["episode"])

    reward_stack.plot(kind="bar", stacked=True, colormap="Paired", figsize=(14, 6), color = ["skyblue", "orange"])
    plt.title("Stacked Episode Reward: Dropout + Clean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_comparison_dropout.png")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y="reward_diff", palette="coolwarm")
    plt.title("Reward Difference (Dropout - Clean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/dropout_reward_difference_bar_plot.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_dropout"]])
    plt.title("Cost Distribution: Clean vs Droput")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/dropout_cost_boxplot.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_dropout"]])
    plt.title("Reward Distribution: Clean vs Dropout")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/dropout_reward_boxplot.png")



def visualize_performance_delay(results_file_path: str):
    df = pd.read_csv(results_file_path)
    df["episode"] = df["episode"].astype(int)

    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Delay": df["reward_delay"]
    }, index=df["episode"])
    reward_stack.plot(kind="bar", stacked=True, colormap="Accent", figsize=(14, 6))
    plt.title("stacked Episode Rewards: Clean + Delayed")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_delay_comparison.png")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y="reward_diff", palette="Spectral")
    plt.title("Reward Difference (Delayed - Clean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_diff_bar_delay.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_delay"]])
    plt.title("Cost Distribution: Clean vs Delay")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_boxplot_delay.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_delay"]])
    plt.title("Reward Distribution: Clean vs Delay")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_boxplot_delay.png")



def visualize_performance(
        results_file_path: str
):
    df = pd.read_csv(results_file_path)
    df["episode"] = df["episode"].astype(int)

    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Noisy": df["reward_noisy"]
    }, index=df["episode"])

    reward_stack.plot(kind="bar", stacked=True, colormap="Paired", figsize=(14, 6))
    plt.title("Stacked Episode Rewards: Clean + Noisy")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_comparison.png")

    # Plot reward difference
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y="reward_diff", palette="coolwarm")
    plt.title("Reward Difference (Noisy - Clean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_difference_bar.png")

    # Cost boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_noisy"]])
    plt.title("Cost Distribution: Clean vs Noisy")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_boxplot.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_noisy"]])
    plt.title("Reward Distribution: Clean vs Noisy")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_boxplot_noise.png")


def visualize_landing_success_rate(
        normal_results_file_path: str,
        gnoise_results_file_path: str,
        obs_delay_results_file_path: str,
        wind_results_file_path: str,
        dropout_results_file_path: str,
        bias_drift_results_file_path: str,
        clipping_results_file_path: str,
        save_path: str = "deepRL_for_autonomous_drones/pics/landing_success_rates.png"
):
    def get_rate(df):
        if "landing_success_rate" in df.columns:
            return df["landing_success_rate"].mean()
        return df[df["step"] == "final"]["landed"].mean() * 100

    normal_df = pd.read_csv(normal_results_file_path)
    noise_df = pd.read_csv(gnoise_results_file_path)
    delay_df = pd.read_csv(obs_delay_results_file_path)
    wind_df = pd.read_csv(wind_results_file_path)
    dropout_df = pd.read_csv(dropout_results_file_path)
    bias_drift_df = pd.read_csv(bias_drift_results_file_path)
    clipping_df = pd.read_csv(clipping_results_file_path)

    rates = {
        "Clean": get_rate(normal_df),
        "Gaussian Noise": get_rate(noise_df),
        "Obs Delay": get_rate(delay_df),
        "Wind": get_rate(wind_df),
        "Dropout": get_rate(dropout_df),
        "Bias Drift": get_rate(bias_drift_df),
        "Clipping": get_rate(clipping_df)
    }

    plt.figure(figsize=(14, 6))
    bars = plt.bar(rates.keys(), rates.values(),
                   color=["skyblue", "orange", "green", "red", "purple", "brown", "teal"])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.1f}%', ha="center", va="bottom", fontsize=10)

    plt.ylim(0, 110)
    plt.ylabel("Landing Success Rate (%)")
    plt.title("Landing Success Rate under Different Observation Disturbances")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Landing success rate saved to {save_path}")




def visualize_performance_wind(
        results_file_path: str
):
    df = pd.read_csv(results_file_path)

    # Reward Difference Line Plot
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="wind_level", y="reward_diff", marker="o")
    plt.title("Reward Difference (Wind - Clean) across Wind Levels")
    plt.xlabel("Wind Level")
    plt.ylabel("Reward Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_diff_wind_normal.png")

    # Cost Difference Line Plot
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="wind_level", y="cost_diff", marker="o")
    plt.title("Cost Difference (Wind - Clean) across Wind Levels")
    plt.xlabel("Wind Level")
    plt.ylabel("Cost Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_diff_wind_normal.png")

    # Landing Success Rate Line Plot
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="wind_level", y="landing_rate_wind", marker="o", label="Wind")
    plt.axhline(df["landing_rate_clean"].iloc[0], color="gray", linestyle="--", label="Clean")
    plt.title("Landing Success Rate across Wind Levels")
    plt.xlabel("Wind Level")
    plt.ylabel("Landing Success Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/landing_rate_vs_wind.png")

    # Reward Stacked Bar Plot
    reward_stack = pd.DataFrame({
        "Clean": df["reward_clean"],
        "Wind": df["reward_wind"]
    }, index=df["wind_level"])

    reward_stack.plot(kind="bar", stacked=True, colormap="Pastel1", figsize=(12, 6))
    plt.title("Stacked Episode Rewards: Clean + Wind")
    plt.xlabel("Wind Level")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_stacked_bar_wind_comparison.png")

    # Cost Stacked Bar Plot
    cost_stack = pd.DataFrame({
        "Clean": df["cost_clean"],
        "Wind": df["cost_wind"]
    }, index=df["wind_level"])

    cost_stack.plot(kind="bar", stacked=True, colormap="Pastel2", figsize=(12, 6))
    plt.title("Stacked Episode Costs: Clean + Wind")
    plt.xlabel("Wind Level")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_stacked_bar_wind_comparison.png")

    # Reward Box Plot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["reward_clean", "reward_wind"]])
    plt.title("Reward Distribution: Clean vs Wind")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/reward_boxplot_wind.png")

    # Cost Box Plot
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", "cost_wind"]])
    plt.title("Cost Distribution: Clean vs Wind")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/cost_boxplot_wind.png")


if __name__ == "__main__":
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-high_wind/checkpoint/ppo_model_wind_high_wind.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-none/checkpoint/ppo_model_wind_none.zip"
    orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-light_breeze/checkpoint/ppo_model_wind_light_breeze.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-light_wind/checkpoint/ppo_model_wind_light_wind.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-medium_wind/checkpoint/ppo_model_wind_medium_wind.zip"

    evaluate_agent_with_wind_effect(
        model_path= orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        render= False,
        episodes=20,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_wind_effect.csv" 
    )
    evaluate_agent_with_gussian_noise(
        model_path=orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        render= False,
        episodes=20, 
        noise_std=0.1,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_gussian_noise_results.csv"
    )
    evaluate_agent_without_noise(
        model_path=orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        episodes=20, 
        render=False,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv"
    )
    evaluate_agent_with_observation_delay(
        model_path=orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        episodes=20,
        delay_frames=1,
        render=False,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_delay_results.csv"
    )
    evaluate_agent_with_dropout_effect(
        model_path=orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        render=False,
        episodes=20,
        dropout_prob=0.1, 
        min_scale=0.01, 
        max_scale=0.1, 
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_dropout_effect.csv"
    )
    compare_with_dropout_results(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        dropout_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_dropout_effect.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_dropout.csv"
    )
    compare_with_and_without_noise_results(
        normal_results_file_path= "deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        noisy_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_gussian_noise_results.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparasion_btw_with_and_without_gussian_noise.csv"
    )
    compare_with_delay_results(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        delay_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_delay_results.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_delay.csv"
    )
    compare_with_and_without_wind_results(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        wind_effects_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_wind_effect.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_and_without_wind_effect.csv"
    )
    visualize_performance(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparasion_btw_with_and_without_gussian_noise.csv"
    )
    visualize_performance_delay(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_delay.csv"
    )
    visualize_performance_wind(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_and_without_wind_effect.csv"
    )
    visualize_performance_dropout(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_dropout.csv"
    )

    evaluate_agent_with_bias_drift(
   model_path=orginal_model_path, 
    env_name="SafetyDroneLanding-v0",
    render=False,
    episodes=20,
    drift_magnitude=0.1,
    save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_bias_drift.csv"
    )

    compare_with_bias_drift_results(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        bias_drift_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_bias_drift.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_bias_drift.csv"
    )

    visualize_performance_bias(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_bias_drift.csv"
    )

    evaluate_agent_with_clipping(
        model_path=orginal_model_path, 
        env_name="SafetyDroneLanding-v0",
        render=False,
        episodes=20,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_clipping.csv"
    )

    compare_with_clipping_results(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        clipping_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_clipping.csv",
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_clipping.csv"
    )

    visualize_performance_clipping(
        results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_comparison_with_clipping.csv"
    )

    visualize_landing_success_rate(
        normal_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_without_noise_results.csv",
        gnoise_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_gussian_noise_results.csv",
        obs_delay_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_obs_delay_results.csv",
        wind_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_wind_effect.csv",
        dropout_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_dropout_effect.csv",
        bias_drift_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_bias_drift.csv",
        clipping_results_file_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_clipping.csv",
        save_path="deepRL_for_autonomous_drones/pics/landing_success_rates.png"
    )



