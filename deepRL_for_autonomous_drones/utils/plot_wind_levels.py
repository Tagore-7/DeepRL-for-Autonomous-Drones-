import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rliable.library import get_interval_estimates
import rliable.metrics as metrics


# ---- x-axis: "speed" m/s   |   "scale" normalised 0–1 ----#
x_axis_mode = "speed"  # "speed"   or   "scale"

df = pd.read_csv("deepRL_for_autonomous_drones/evaluation/evaluations/wind_results.csv")

speed_map = {"none": 0.0, "light": 4.47, "medium": 8.94, "high": 17.88}
df["wind_speed"] = df["wind_level"].map(speed_map)
df = df.sort_values("wind_speed")

if x_axis_mode == "speed":
    x_vals = df["wind_speed"].values
    x_label = "Wind speed (m/s)"
elif x_axis_mode == "scale":
    x_vals = df["wind_speed"].values / speed_map["high"]
    x_label = "Wind scale (0 = calm, 1 = 17.9 m/s)"
else:
    raise ValueError("x_axis_mode must be 'speed' or 'scale'")

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
ax1.plot(x_vals, reward_mean, label="Reward", color="tab:blue")
ax1.fill_between(x_vals, reward_lower, reward_upper, alpha=0.3, color="tab:blue")
ax1.set_ylabel("Reward")
ax1.grid(True, linestyle="--", alpha=0.4)

# ---- Cost plot ----#
ax2.plot(x_vals, cost_mean, label="Cost", color="tab:red")
ax2.fill_between(x_vals, cost_lower, cost_upper, alpha=0.3, color="tab:red")
ax2.set_ylabel("Cost")
ax2.set_xlabel(x_label)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
