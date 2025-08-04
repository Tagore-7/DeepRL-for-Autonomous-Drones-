import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import ast
import os


# Load trajectory data
TRAJ_PATH = "deepRL_for_autonomous_drones/evaluation/evaluations/eval_trajectories.csv"
df = pd.read_csv(TRAJ_PATH)

if 'action' not in df.columns:
    print("ERROR: 'action' column missing. Check data saving logic.")
    exit(1)

if isinstance(df["action"].iloc[0], str):
    df["action"] = df["action"].apply(ast.literal_eval)

action_array = np.vstack(df["action"])
action_df = pd.DataFrame(action_array, columns=[f"action_{i}" for i in range(action_array.shape[1])])
df = pd.concat([df, action_df], axis=1)

# Compute velocities if not present
if all(col in df.columns for col in ['x', 'y', 'z']):
    dt = 0.1  # Time step (10Hz simulation)
    for axis in ['x', 'y', 'z']:
        df[f'v{axis}'] = df.groupby('episode')[axis].diff() / dt
    df[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']].fillna(0)



def plot_3d_trajectories(df):
    """Plot 3D trajectories for both successful and failed landings"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot successful trajectories
    success_df = df[df["landed"] == True]
    for ep_id in success_df["episode"].unique()[:3]:
        ep_data = success_df[success_df["episode"] == ep_id]
        ax.plot(ep_data["x"], ep_data["y"], ep_data["z"], 
                color='green', alpha=0.7, linewidth=2, 
                label='Success' if ep_id == success_df["episode"].iloc[0] else "")
    
    # Plot failed trajectories
    failed_df = df[df["landed"] == False]
    for ep_id in failed_df["episode"].unique()[:5]:
        ep_data = failed_df[failed_df["episode"] == ep_id]
        ax.plot(ep_data["x"], ep_data["y"], ep_data["z"], 
                color='red', alpha=0.7, linewidth=2,
                label='Failure' if ep_id == failed_df["episode"].iloc[0] else "")
    
    # Plot landing pad
    ax.scatter([0], [0], [0], s=200, c='blue', marker='X', label='Landing Pad')
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Altitude")
    ax.set_title("Success vs Failure Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/3d_trajectory_comparison.png")
    plt.close()
    print("Saved 3D trajectory comparison plot")

def plot_state_time_series(ep_data, success=True):
    """Plot state variables and actions over time for an episode"""
    plt.figure(figsize=(12, 8))
    
    # Position over time
    plt.subplot(2, 1, 1)
    plt.plot(ep_data["step"], ep_data["x"], label="X Position")
    plt.plot(ep_data["step"], ep_data["y"], label="Y Position")
    plt.plot(ep_data["step"], ep_data["z"], label="Z Altitude")
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel("Position")
    plt.legend()
    status = "Success" if success else "Failure"
    plt.title(f"State Variables Over Time - {status} Episode {ep_data['episode'].iloc[0]}")
    
    # Actions over time
    plt.subplot(2, 1, 2)
    action_cols = [col for col in ep_data.columns if col.startswith('action_')]
    for i, col in enumerate(action_cols):
        plt.plot(ep_data["step"], ep_data[col], label=f"Action {i}")
    plt.xlabel("Time Step")
    plt.ylabel("Action Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"deepRL_for_autonomous_drones/pics/time_series_ep{ep_data['episode'].iloc[0]}_{status}.png")
    plt.close()

def plot_phase_space(df):
    """Create phase space plots for key variables"""
    plt.figure(figsize=(12, 10))
    
    # Velocity vs Altitude
    if all(col in df.columns for col in ['z', 'vz']):
        plt.subplot(2, 2, 1)
        plt.scatter(df["z"], df["vz"], c=df["step"], cmap='viridis', alpha=0.6)
        plt.colorbar(label="Time Step")
        plt.xlabel("Altitude (Z)")
        plt.ylabel("Vertical Velocity (Vz)")
        plt.title("Altitude vs Vertical Velocity")
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # XY Position
    plt.subplot(2, 2, 2)
    plt.scatter(df["x"], df["y"], c=df["z"], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label="Altitude (Z)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')
    plt.title("XY Position Colored by Altitude")
    plt.scatter(0, 0, s=100, c='red', marker='X')
    
    # Action vs Altitude
    if 'z' in df.columns:
        plt.subplot(2, 2, 3)
        action_cols = [col for col in df.columns if col.startswith('action_')][:4]
        for i, col in enumerate(action_cols):
            plt.scatter(df["z"], df[col], alpha=0.3, label=f"Action {i}")
        plt.xlabel("Altitude (Z)")
        plt.ylabel("Action Value")
        plt.legend()
        plt.title("Actions vs Altitude")
    
    # Velocity distribution
    if all(col in df.columns for col in ['vx', 'vy']):
        plt.subplot(2, 2, 4)
        plt.scatter(df["vx"], df["vy"], alpha=0.3)
        plt.title("Low-Variance Velocity Data")
        plt.xlabel("X Velocity (Vx)")
        plt.ylabel("Y Velocity (Vy)")
        # plt.title("Horizontal Velocity Distribution")
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/phase_space_analysis.png")
    plt.close()
    print("Saved phase space analysis plot")

def plot_failure_causes(failed_df):
    """Categorize and visualize failure root causes"""
    if "failure_type" not in failed_df.columns:
        print("No failure_type column found. Skipping root cause analysis.")
        return failed_df

    plt.figure(figsize=(10, 6))
    sns.countplot(data=failed_df, y="failure_type", 
                  order=failed_df["failure_type"].value_counts().index,
                  palette="Reds_r")
    plt.xlabel("Count")
    plt.ylabel("Failure Type")
    plt.title("Failure Root Cause Analysis (Recorded Labels)")
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/failure_causes.png")
    plt.close()
    print("Saved failure root cause analysis plot")
    
    return failed_df


def plot_enhanced_heatmap(failed_df):
    """Create enhanced heatmaps of failure locations"""
    fig = plt.figure(figsize=(14, 6))
    
    # XY Heatmap
    plt.subplot(1, 2, 1)
    plt.hist2d(failed_df["x"], failed_df["y"], bins=30, cmap="Reds")
    plt.colorbar(label="Failure Density")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Failure Density (XY Plane)")
    plt.scatter(0, 0, s=100, c='blue', marker='X')
    
    # XZ Heatmap
    if 'z' in failed_df.columns:
        plt.subplot(1, 2, 2)
        plt.hist2d(failed_df["x"], failed_df["z"], bins=30, cmap="Blues")
        plt.colorbar(label="Failure Density")
        plt.xlabel("X Position")
        plt.ylabel("Altitude (Z)")
        plt.title("Failure Density (XZ Plane)")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/enhanced_failure_heatmap.png")
    plt.close()
    print("Saved enhanced failure heatmap")

def plot_action_profiles(success_df, failed_df):
    """Compare action profiles between success and failure cases"""
    plt.figure(figsize=(14, 8))
    
    # Action distributions for successful episodes
    plt.subplot(2, 1, 1)
    action_cols = [col for col in success_df.columns if col.startswith('action_')]
    success_actions = success_df[action_cols].mean()
    plt.bar(range(len(action_cols)), success_actions.values, color='green', alpha=0.7)
    plt.xticks(range(len(action_cols)), [f"Action {i}" for i in range(len(action_cols))])
    plt.ylabel("Mean Action Value")
    plt.title("Successful Landing Action Profile")
    
    # Action distributions for failed episodes
    plt.subplot(2, 1, 2)
    fail_actions = failed_df[action_cols].mean()
    plt.bar(range(len(action_cols)), fail_actions.values, color='red', alpha=0.7)
    plt.xticks(range(len(action_cols)), [f"Action {i}" for i in range(len(action_cols))])
    plt.ylabel("Mean Action Value")
    plt.title("Failed Landing Action Profile")
    
    plt.tight_layout()
    plt.savefig("deepRL_for_autonomous_drones/pics/action_profile_comparison.png")
    plt.close()
    print("Saved action profile comparison plot")

def plot_trajectory_analysis(df):
    """Main function to run all analysis plots"""
    # Filter datasets
    success_df = df[df["landed"] == True]
    failed_df = df[df["landed"] == False]
    
    # Plot 3D trajectories
    plot_3d_trajectories(df)
    
    # Plot failure causes if we have failures
    if not failed_df.empty:
        failed_df = plot_failure_causes(failed_df)
        plot_enhanced_heatmap(failed_df)
        plot_phase_space(failed_df)
        
        # Plot time series for first 3 failures
        for ep_id in failed_df["episode"].unique()[:3]:
            ep_data = failed_df[failed_df["episode"] == ep_id]
            plot_state_time_series(ep_data, success=False)
    
    # Plot success trajectories if available
    if not success_df.empty:
        # Plot time series for first 3 successes
        for ep_id in success_df["episode"].unique()[:3]:
            ep_data = success_df[success_df["episode"] == ep_id]
            plot_state_time_series(ep_data, success=True)
        
        # Compare action profiles
        if not failed_df.empty:
            plot_action_profiles(success_df, failed_df)


print("Starting trajectory analysis...")
plot_trajectory_analysis(df)
print("All analysis plots saved to deepRL_for_autonomous_drones/pics/")