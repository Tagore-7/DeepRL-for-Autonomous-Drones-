import pandas as pd
from pandas._config import config as _pdconf
from pandas._config.config import OptionError
import matplotlib
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position, disable=ungrouped-imports
import seaborn as sns  # pylint: disable=wrong-import-position

try:
    pd.set_option("mode.use_inf_as_null", False)
except OptionError:
    _pdconf.register_option("mode.use_inf_as_null", False)


def format_name(key: str) -> str:
    """Turn 'noise_0.1' -> 'Action noise (σ = 0.1)', 'wind_medium' -> 'Wind (medium)'."""
    if key.startswith("noise_"):
        sigma = key.split("_", 1)[1]
        return f"Action noise (σ = {sigma})"
    if key.startswith("wind_"):
        lvl = key.split("_", 1)[1]
        return f"Wind ({lvl})"
    if key.startswith("delay_"):
        k = key.split("_", 1)[1]
        return f"Action delay ({k} steps)"
    if key.startswith("fail_"):
        _, rest = key.split("fail_", 1)
        idx_str, steps_str = rest.split("_", 1)
        return f"Rotor failure (rotor {idx_str}, {steps_str} steps)"
    if key.startswith("flip_"):
        idxs = key.split("_", 1)[1].replace(",", ", ")
        return f"Sign flip (rotor {idxs})"
    return key.capitalize()


def plot_combined_metric(
    summary: pd.DataFrame,
    variant_keys: list[str],
    metric_prefix: str,  # "reward_" or "cost_"
    out_path: Path,
):
    melt = pd.melt(
        summary,
        id_vars=["episode"],
        value_vars=[f"{metric_prefix}{k}" for k in variant_keys],
        var_name="variant",
        value_name="value",
    )
    melt["formatted"] = melt["variant"].map(lambda v: format_name(v[len(metric_prefix) :]))

    n = len(variant_keys)
    plt.figure(figsize=(8, max(6, n * 0.25)))

    sns.boxplot(
        data=melt,
        y="formatted",
        x="value",
        fliersize=0,
        linewidth=0.5,
        width=0.5,
        color="lightgrey",
    )
    sns.stripplot(
        data=melt,
        y="formatted",
        x="value",
        size=2,
        color="black",
        alpha=0.8,
        jitter=0.15,
    )
    title = textwrap.fill(f"{metric_prefix.capitalize()}Distribution Across Disruptor Combinations".replace("_", " "), width=35)
    plt.title(title)
    plt.xlabel(metric_prefix.capitalize().rstrip("_"))
    plt.ylabel("Disruptor Combination")
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_combined_success(success: Dict[str, float], out_path: Path):
    items = sorted(success.items(), key=lambda kv: kv[1], reverse=True)
    formatted_labels = [format_name(k) for k, _ in items]
    rates = [v for _, v in items]

    n = len(formatted_labels)
    plt.figure(figsize=(8, max(6, n * 0.25)))
    sns.barplot(y=formatted_labels, x=rates, orient="h")
    plt.xlim(0, 1.05)
    plt.xlabel("Landing Success Rate")
    title = textwrap.fill("Landing Success Rate under Observation Disturbance Combinations", width=35)
    plt.title(title)
    for y, r in enumerate(rates):
        plt.text(r + 0.01, y, f"{r*100:.0f}%", va="center")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_failure_causes(fail_list: list[str], out: Path):
    counts = pd.Series(fail_list).value_counts()
    order = ["Hard landing", "Plane crash", "Tree crash", "Timeout", "Out-of-bounds", "Other"]
    counts = counts.reindex(order, fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=counts.index, x=counts.values, palette="Reds_r")
    plt.title("Failure Root Cause Analysis")
    plt.xlabel("Count")
    plt.ylabel("Failure Type")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_action_profiles(summary: pd.DataFrame, out: Path):
    succ = summary[summary["failure"] == "SUCCESS"]
    fail = summary[summary["failure"] != "SUCCESS"]
    means_succ = succ[[f"act_m{i}" for i in range(4)]].mean()
    means_fail = fail[[f"act_m{i}" for i in range(4)]].mean()

    actions = [f"Action {i}" for i in range(4)]
    _fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(actions, means_succ.tolist(), color="forestgreen")
    axes[0].set_title("Successful Landing Action Profile")
    axes[0].set_ylabel("Mean Action Value")
    axes[1].bar(actions, means_fail.tolist(), color="crimson")
    axes[1].set_title("Failed Landing Action Profile")
    axes[1].set_ylabel("Mean Action Value")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _label_format(name: str) -> str:
    """Beautify internal variant keys for plots."""
    if name.startswith("wind_"):
        level = name.split("_", 1)[1]
        return f"Wind ({level})"
    if name.startswith("noise_"):
        sigma = name.split("_", 1)[1]
        return rf"Noise (σ={sigma})"
    if name == "noise":
        return "Action noise"
    return name.capitalize()


def plot_stacked_rewards(df: pd.DataFrame, base: str, variant: str, out: Path):
    plt.figure(figsize=(14, 6))
    reward_stack = pd.DataFrame(
        {
            _label_format(base): df[f"reward_{base}"],
            _label_format(variant): df[f"reward_{variant}"],
        },
        index=df["episode"],
    )
    reward_stack.plot(kind="bar", stacked=True, colormap="Paired")
    plt.title(f"Stacked Episode Rewards: {_label_format(base)} + {_label_format(variant)}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_reward_diff(df: pd.DataFrame, base: str, variant: str, out: Path):
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="episode", y=f"reward_diff_{variant}", palette="coolwarm")
    plt.title(f"Reward Difference ({_label_format(variant)} - {_label_format(base)})")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_cost_box(df: pd.DataFrame, variant: str, out: Path):
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["cost_clean", f"cost_{variant}"]])
    plt.title(f"Cost Distribution: Clean vs {_label_format(variant)}")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_success_rates(success: Dict[str, float], out_path: Path):
    labels, rates = list(success.keys()), list(success.values())
    width = max(6, len(labels) * 1.2)  # widen figure for many labels
    plt.figure(figsize=(width, 6))
    rects = plt.bar(labels, rates)
    plt.ylim(0, 1.05)
    plt.ylabel("Landing Success Rate")
    plt.title("Landing Success Rate under Disturbances")
    plt.xticks(rotation=20, ha="right")
    for rect, rate in zip(rects, rates):
        h = rect.get_height()
        plt.annotate(f"{rate*100:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
