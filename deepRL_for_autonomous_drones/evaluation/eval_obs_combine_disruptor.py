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
import itertools




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

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        env.unwrapped.landed = False
        done = False
        delay_frames = 2
        delay_counter = 0
        delayed_obs = obs.copy()

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

            if "delay" in combination:
                delay_counter += 1
                if delay_counter >= delay_frames:
                    delayed_obs = obs.copy()
                    delay_counter = 0
            else:
                delayed_obs = obs.copy()


        if hasattr(env.unwrapped, "landed") and env.unwrapped.landed:
            landings += 1

    success_rate = landings / episodes * 100

    return {
        "combination": "+".join(combination),
        "reward": rewards / episodes,
        "cost": costs / episodes,
        "landing_success_rate": success_rate
    }


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
    df_sorted = df.sort_values("landing_success_rate", ascending=False)

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




def run_all_combinations(model_path: str, env_name: str, save_path: str, render: bool = False, episodes: int = 20):
    base_disruptors = ["gaussian", "dropout", "clipping", "bias", "delay"]
    wind_options = ["none","light_breeze", "light_wind", "medium_wind", "high_wind"]
    all_results = []
    record_stats = []

    for wind in wind_options:
        for r in range(7):  # 1-way to 6-way combinations
            for combo in itertools.combinations(base_disruptors, r):
                full_combo = list(combo) + [wind]
                result = evaluate_combination(full_combo, model_path, env_name, episodes, render)
                all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"Saved all combination results to {save_path}")  

    # # plot the cost and reward boxplot for all combinations 
    plot_boxplot(save_path)
    
    plot_combined_landing_rates(save_path)


if __name__ == "__main__":
    run_all_combinations(
        model_path="deepRL_for_autonomous_drones/training/results/fast-safe-rl/SafetyDroneLanding-v0-cost-0/ppo-370b/checkpoint/ppo_model.zip",
        env_name="SafetyDroneLanding-v0",
        render=False,
        episodes=20,
        save_path="deepRL_for_autonomous_drones/evaluation/evaluations/eval_results_with_all_combinations.csv"
    )


