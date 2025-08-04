import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import ast

# Load trajectory data
TRAJ_PATH = "deepRL_for_autonomous_drones/evaluation/evaluations/eval_trajectories.csv"
df = pd.read_csv(TRAJ_PATH)

if 'action' not in df.columns:
    print("ERROR: 'action' column missing. Check data saving logic.")
    exit(1)

# Parse stringified actions if needed
if isinstance(df["action"].iloc[0], str):
    df["action"] = df["action"].apply(ast.literal_eval)

# Flatten actions into separate columns
action_array = np.vstack(df["action"])
action_df = pd.DataFrame(action_array, columns=[f"action_{i}" for i in range(action_array.shape[1])])
df = pd.concat([df, action_df], axis=1)

# Filter failed episodes
failed_df = df[df["landed"] == False]

# --- 1. Plot 3D Trajectories of a Few Failed Episodes ---
for ep_id in failed_df["episode"].unique()[:5]:
    ep_data = failed_df[failed_df["episode"] == ep_id]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ep_data["x"], ep_data["y"], ep_data["z"], label=f"Episode {ep_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Trajectory of Failed Episode {ep_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"deepRL_for_autonomous_drones/pics/trajectory_3d_ep{ep_id}.png")
    plt.close()

# --- 2. Top-Down XY Trajectories of Failed Episodes ---
for ep_id in failed_df["episode"].unique()[:5]:
    ep_data = failed_df[failed_df["episode"] == ep_id]
    plt.figure(figsize=(6,6))
    plt.plot(ep_data["x"], ep_data["y"], marker='o', label=f"Ep {ep_id}")
    plt.title(f"Top-Down Trajectory of Failed Ep {ep_id}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"deepRL_for_autonomous_drones/pics/trajectory_xy_ep{ep_id}.png")
    plt.close()

# --- 3. Violin Plot of Action Distribution for Failed Episodes ---
plt.figure(figsize=(10,6))
sns.violinplot(data=failed_df[[f"action_{i}" for i in range(action_array.shape[1])]])
plt.title("Action Distribution in Failed Episodes")
plt.ylabel("Action Value")
plt.xlabel("Action Dimension")
plt.tight_layout()
plt.savefig("deepRL_for_autonomous_drones/pics/failed_action_distribution.png")
plt.close()

# --- 4. Heatmap of Failupltre Zones in XY Plane ---
plt.figure(figsize=(8,6))
plt.hist2d(failed_df["x"], failed_df["y"], bins=30, cmap="Reds")
plt.colorbar(label="Failure Density")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Failure Density Heatmap (X vs Y)")
plt.tight_layout()
plt.savefig("deepRL_for_autonomous_drones/pics/failure_density_heatmap.png")
plt.close()

print("All failure trajectory and action plots saved.")
