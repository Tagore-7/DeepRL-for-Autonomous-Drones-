import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rliable.library import get_interval_estimates
import rliable.metrics as metrics

df = pd.read_csv("../evaluation/evaluations/wind_results.csv")

# ---- Wind levels = tasks ----#
wind_levels = df["wind"].unique()
num_tasks = len(wind_levels)

# ---- Matrix shape [num_runs, num_tasks], only 1 for now ----#
reward_matrix = np.expand_dims(df["reward"].values, axis=0)
cost_matrix = np.expand_dims(df["cost"].values, axis=0)

agg_fn = lambda x: x.squeeze()
point_estimates, interval_estimates = get_interval_estimates(
    score_dict={"reward": reward_matrix, "cost": cost_matrix},
    func=agg_fn,
    reps=5000,
    method="percentile",
    confidence_interval_size=0.95,
)

# ---- Extract point + CI ----#
reward_mean = point_estimates["reward"]
reward_lower, reward_upper = interval_estimates["reward"]
cost_mean = point_estimates["cost"]
cost_lower, cost_upper = interval_estimates["cost"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
fig.suptitle("Wind Effect on Reward/Cost", fontsize=14)

# ---- Reward plot ----#
ax1.plot(wind_levels, reward_mean, label="Reward", color="tab:blue")
ax1.fill_between(wind_levels, reward_lower, reward_upper, alpha=0.3, color="tab:blue")
ax1.set_ylabel("Reward")
ax1.grid(True, linestyle="--", alpha=0.4)

# ---- Cost plot ----#
ax2.plot(wind_levels, cost_mean, label="Cost", color="tab:red")
ax2.fill_between(wind_levels, cost_lower, cost_upper, alpha=0.3, color="tab:red")
ax2.set_ylabel("Cost")
ax2.set_xlabel("Wind Scale")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
