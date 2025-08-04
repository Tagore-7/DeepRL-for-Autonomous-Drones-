import numpy as np
import pandas as pd 
import torch 
import os 
import deepRL_for_autonomous_drones
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import seaborn as sns 
import matplotlib.pyplot as plt 
from deepRL_for_autonomous_drones import envs
import itertools
import pybullet as p





class ObservationDisruptor:
    def __init__(self, obs: np.ndarray = None):
        self.obs = obs

    def applyGussianDisruptor(self, noise_std):
        noise = np.random.normal(0.0, noise_std, size=self.obs.shape)
        noisy_obs = self.obs + noise
        return self.obs, noise, noisy_obs
    
    def dropoutDisruptor(self,  dropout_prob: float = 0.3, min_scale: float = 0.1, max_scale: float = 0.3):
        mask = np.random.binomial(1, 1 - dropout_prob, size = self.obs.shape)
        scale_factors = np.random.uniform(min_scale, max_scale, size = self.obs.shape)
        dropped_obs = self.obs * (mask + ( 1- mask) * scale_factors)
        return self.obs, mask, dropped_obs
    
    def biasDriftDisruptor(self,  drift_magnitude: float = 0.2, drift_type: str = "constant"):
        if drift_type == "constant":
            bias = np.ones_like(self.obs) * drift_magnitude
        elif drift_type == "random_walk":
            bias = np.random.normal(0.0, drift_magnitude, size=self.obs.shape)
        biased_obs = self.obs + bias 
        return self.obs, bias, biased_obs
    
    def clippingDisruptor(self, clip_min: float = -2, clip_max: float = 7):
        original_obs = self.obs.copy()
        clipped_obs = np.clip(original_obs, clip_min, clip_max)
        return self.obs, clipped_obs

def make_env(env_name: str, wind_level: str = "none", render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = FlattenObservation(env)
    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(wind_level != "none")
    if hasattr(env.unwrapped, "setWindLevel"):
        env.unwrapped.setWindLevel(wind_level)
    return env



def evaluate_combination(combination, model_path, env_name, episodes, render):
    wind_levels = {
        "none": 0.0,
        "light_breeze": 2.24,
        "light_wind": 4.47,
        "medium_wind": 8.94,
        "high_wind": 17.88
    }
    wind_label = next((item for item in combination if item in wind_levels), "none")
    env = make_env(env_name, wind_level=wind_label, render=render)
    model = PPO.load(model_path, device="cpu")

    rewards, costs, landings = 0, 0, 0
    step_records = []

    for ep in range(episodes):
        step_records_ep = []
        obs, _ = env.reset(seed=ep)
        env.unwrapped.landed = False
        done = False
        delay_frames = 2
        delay_counter = 0
        delayed_obs = obs.copy()
        step_count = 0

        while not done:
            disrupted_obs = delayed_obs.copy()

            if "gaussian" in combination:
                _, _, disrupted_obs = ObservationDisruptor(disrupted_obs).applyGussianDisruptor(noise_std=0.05)
            if "dropout" in combination:
                _, _, disrupted_obs = ObservationDisruptor(disrupted_obs).dropoutDisruptor(dropout_prob=0.3)
            if "bias" in combination:
                _, _, disrupted_obs = ObservationDisruptor(disrupted_obs).biasDriftDisruptor(drift_magnitude=0.1)
            if "clipping" in combination:
                _, disrupted_obs = ObservationDisruptor(disrupted_obs).clippingDisruptor(clip_min=-2, clip_max=7)


            action, _ = model.predict(disrupted_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward
            costs += info.get("cost", 0.0)

            record = {
                "episode": ep,
                "step": step_count,
                "combination": "+".join(combination),
                "x": env.unwrapped.drone.getDroneStateVector()[0],
                "y": env.unwrapped.drone.getDroneStateVector()[1],
                "z": env.unwrapped.drone.getDroneStateVector()[2],
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "reward": reward,
                "cost": info.get("cost", 0.0),
                "done": done,
                "landed": env.unwrapped.landed
            }
            step_records_ep.append(record)
            
            step_count += 1

            if "delay" in combination:
                delay_counter += 1
                if delay_counter >= delay_frames:
                    delayed_obs = obs.copy()
                    delay_counter = 0
            else:
                delayed_obs = obs.copy()


        if hasattr(env.unwrapped, "landed") and env.unwrapped.landed:
            landings += 1
        
        # Determine failure reason after episode ends
        failure_lbl = "None"
        if not env.unwrapped.landed:
            drone_id = env.unwrapped.drone.getDroneID()
            plane_id = env.unwrapped.plane

            plane_hits = p.getContactPoints(bodyA=drone_id, bodyB=plane_id)
            tree_hits = any(p.getContactPoints(bodyA=drone_id, bodyB=t) for t in env.unwrapped.trees)

            if plane_hits:
                failure_lbl = "Plane crash"
            elif tree_hits:
                failure_lbl = "Tree crash"
            elif truncated:
                failure_lbl = "Timeout"
            elif abs(env.unwrapped.drone.get_position()[2]) > env.unwrapped.MAX_Z:
                failure_lbl = "Out-of-bounds"
            else:
                vz = env.unwrapped.drone.get_linear_velocity()[2]
                failure_lbl = "Hard landing" if abs(vz) > 0.5 else "Other"
        else:
            failure_lbl = "None"

        
        for record in step_records_ep:
            record["failure_type"] = failure_lbl
        
        step_records.extend(step_records_ep)


    success_rate = landings / episodes * 100

    return {
        "combination": "+".join(combination),
        "reward": rewards / episodes,
        "cost": costs / episodes,
        "landing_success_rate": success_rate,
        "failure_type": failure_lbl,
    }, step_records


def plot_boxplot(results_file_path: str):
    df = pd.read_csv(results_file_path)
    df = df.sort_values("landing_success_rate", ascending=False)

    # Generate a random color for each combination
    unique_combinations = df["combination"].unique()
    palette = {combo: np.random.rand(3,) for combo in unique_combinations}

    # Reward boxplot
    plt.figure(figsize=(15, 20))
    sns.boxplot(data=df, y="combination", x="reward", palette=palette)
    plt.title("Reward Distribution Across Disruptor Combinations")
    plt.xlabel("Reward")
    plt.ylabel("Disruption Combination")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/combined_reward_boxplot.png")
    print("Saved reward boxplot")

    # Cost boxplot
    plt.figure(figsize=(15, 30))
    sns.boxplot(data=df, y="combination", x="cost", palette=palette)
    plt.title("Cost Distribution Across Disruptor Combinations")
    plt.xlabel("Cost")
    plt.ylabel("Disruption Combination")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/combined_cost_boxplot.png")
    print("Saved cost boxplot")


def plot_combined_landing_rates(results_file):
    df = pd.read_csv(results_file)
    # df_sorted = df.sort_values("landing_success_rate", ascending=False)
    df_sorted = df.copy()

    plt.figure(figsize=(15, 30))
    ax = sns.barplot(
        y="combination",
        x="landing_success_rate",
        data=df_sorted,
        palette=np.random.rand(len(df_sorted), 3)
    )

    for i, val in enumerate(df_sorted["landing_success_rate"]):
        ax.text(val + 0.5, i, f"{val:.1f}%", va='center')

    plt.xlabel("Landing Success Rate (%)")
    plt.ylabel("Disruption Combination")
    plt.title("Landing Success Rate under Observation Disturbance Combinations")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/combined_landing_success_rates.png")
    print("Saved updated landing success rate plot")


def plot_average_by_disruptor_type(results_file: str, output_path: str):
    df = pd.read_csv(results_file)
    
    # Define disruptor types (excluding wind levels)
    disruptor_types = ["gaussian", "bias", "dropout", "clipping", "delay"]
    for dtype in disruptor_types:
        df[dtype] = df["combination"].str.contains(dtype)

    avg_success_rates = {}
    
    # Calculate control group (no disruptors)
    control_mask = ~df[disruptor_types].any(axis=1)
    avg_success_rates["control"] = df[control_mask]["landing_success_rate"].mean()
    
    # Calculate averages for each disruptor type
    for dtype in disruptor_types:
        avg_success_rates[dtype] = df[df[dtype]]["landing_success_rate"].mean()

    # Prepare data for plotting
    categories = ["Control"] + [dtype.capitalize() for dtype in disruptor_types]
    values = [avg_success_rates['control']] + [avg_success_rates[d] for d in disruptor_types]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=categories, y=values, palette="viridis")
    
    plt.title("Average Landing Success Rate by Disruptor Type", fontsize=16)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.xlabel("Disruptor Type", fontsize=14)
    plt.ylim(0, 100)
    
    # Add value annotations
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved disruptor type average plot to {output_path}")


def plot_average_by_wind_level(results_file: str, output_path: str):
    df = pd.read_csv(results_file)
    
    # Define wind levels
    wind_levels = ["none", "light_breeze", "light_wind", "medium_wind", "high_wind"]
    
    avg_success_rates = {}
    
    # Calculate averages for each wind level
    for wind in wind_levels:
        # Find combinations containing this wind level
        wind_mask = df["combination"].str.contains(wind)
        avg_success_rates[wind] = df[wind_mask]["landing_success_rate"].mean()

    # Prepare data for plotting
    categories = [wind.replace("_", " ").title() for wind in wind_levels]
    values = [avg_success_rates[d] for d in wind_levels]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=categories, y=values, palette="rocket")
    
    plt.title("Average Landing Success Rate by Wind Level", fontsize=16)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.xlabel("Wind Level", fontsize=14)
    plt.ylim(0, 100)
    
    # Add value annotations
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved wind level average plot to {output_path}")




def run_all_combinations(model_path: str, env_name: str, save_path: str, render: bool = False, episodes: int = 20, record_stats_path: str = None):
    base_disruptors = ["gaussian", "dropout", "clipping", "bias", "delay"]
    wind_options = ["none","light_breeze", "light_wind", "medium_wind", "high_wind"]
    all_results = []
    all_step_logs = []

    for wind in wind_options:
        for r in range(7):  # 1-way to 6-way combinations
            for combo in itertools.combinations(base_disruptors, r):
                full_combo = list(combo) + [wind]
                result, step_logs = evaluate_combination(full_combo, model_path, env_name, episodes, render)
                all_results.append(result)
                all_step_logs.extend(step_logs)

    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"Saved all combination results to {save_path}")  

    if record_stats_path:
        pd.DataFrame(all_step_logs).to_csv(record_stats_path, index=False)
        print(f"Saved step-by-step trajectory stats to {record_stats_path}")


    # # plot the cost and reward boxplot for all combinations 
    plot_boxplot(save_path)
    
    plot_combined_landing_rates(save_path)

    plot_average_by_disruptor_type(
        save_path,
        "deepRL_for_autonomous_drones/pics/disruptor_type_averages.png"
    )
    plot_average_by_wind_level(
        save_path,
        "deepRL_for_autonomous_drones/pics/wind_level_averages.png"
    )

if __name__ == "__main__":
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-high_wind/checkpoint/ppo_model_wind_high_wind.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-none/checkpoint/ppo_model_wind_none.zip"
    orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-light_breeze/checkpoint/ppo_model_wind_light_breeze.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-light_wind/checkpoint/ppo_model_wind_light_wind.zip"
    # orginal_model_path =  "benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-0-wind-medium_wind/checkpoint/ppo_model_wind_medium_wind.zip"


    run_all_combinations(
        model_path=orginal_model_path,
        env_name="SafetyDroneLanding-v0",
        render=False,
        episodes=20,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_all_combinations.csv",
        record_stats_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_trajectories.csv"
    )


