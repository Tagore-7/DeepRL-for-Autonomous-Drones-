import numpy as np
import pandas as pd
import random
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import pyrallis
import torch
import gymnasium as gym
from tianshou.data import Batch
from fsrl.agent import PPOLagAgent
from fsrl.utils import BaseLogger, exp_util
from fsrl.utils.exp_util import seed_all
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import (
    DroneControllerRPM,
)
from deepRL_for_autonomous_drones.utils.action_plotting import (
    format_name,
    plot_stacked_rewards,
    plot_reward_diff,
    plot_cost_box,
    plot_success_rates,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="is_categorical_dtype is deprecated and will be removed",
)


# ---------------------- CONFIGS ---------------------- #
@dataclass
class EvalCfg:
    model_dir: str = "../training/benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-6/ppol-8"
    episodes: int = 20
    render: bool = False
    out_dir: str = "evaluation/action_robustness"

    noise_sigmas: str = "0.05, 0.1, 0.5, 1.0"  # e.g. "0.05,0.1,0.5,1.0" or "none"
    wind_levels: str = "light_breeze, light, medium, high"  # light_breeze,light,medium,high
    delay_steps: str = "1, 2, 4, 5, 10"
    rotor_failures: str = "0:10"  # comma-separated "idx:steps", e.g. "0:10,2:20", or "none"

    run_noise: bool = True
    run_wind: bool = True
    run_delay: bool = True

    def __post_init__(self):
        self.run_noise = self.noise_sigmas.strip().lower() != "none"
        self.run_wind = self.wind_levels.strip().lower() != "none"
        self.run_delay = self.delay_steps.strip().lower() != "none"
        self.run_rotor_fail = self.rotor_failures.strip().lower() != "none"


# ---------------------- ENVS ---------------------- #


def make_env(
    task: str,
    render: bool = False,
    wind_level: str | None = None,
    enable_wind: bool = False,
) -> gym.Env:
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)

    if wind_level is None:
        wind_level = "none"

    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(enable_wind)
    if hasattr(env.unwrapped, "enableWind"):
        env.unwrapped.enableWind(enable_wind)
    if hasattr(env.unwrapped, "setWindLevel"):
        env.unwrapped.setWindLevel(wind_level)

    return env


# ---------------------- HELPERS ---------------------- #


def to_batch(obs):
    if isinstance(obs, dict):
        obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    else:
        obs = np.expand_dims(obs, 0)
    return Batch(obs=obs)


# ---------------------- ROLLOUT ---------------------- #


def rollout(
    policy,
    env: gym.Env,
    n_eps: int,
    noise_sigma: float = 0.0,
    fail_idx: int | None = None,
    fail_steps: int = 0,
    fail_delay_sec: float = 0.0,
):
    """Return lists: reward, cost, len, landed  (per episode)."""
    drone = env.unwrapped.drone

    drone.args.noise_sigma = noise_sigma
    drone.args.enabled = noise_sigma > 0
    ctrl_freq = env.unwrapped.CTRL_FREQ

    rewards, costs, lens, landed = [], [], [], []
    for _ in range(n_eps):

        if fail_idx is not None and fail_steps > 0:
            drone.armRotorFailure(
                idx=fail_idx,
                steps=fail_steps,
                delay_sec=fail_delay_sec,
                ctrl_freq=ctrl_freq,
            )
        else:
            drone.resetRotorFailure()

        seed = random.randint(0, 100)
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        ep_r = ep_c = 0.0
        while not (done or trunc):
            with torch.no_grad():
                act = policy(to_batch(obs), deterministic=True).act.squeeze(0)
            obs, r, done, trunc, info = env.step(act)
            # print(f"|OBS| - {obs}")
            pos = env.unwrapped.drone.getDroneStateVector()[:3]
            print(f"|POS|: {pos}")
            ep_r += r
            ep_c += info.get("cost", 0.0)
        rewards.append(float(ep_r))
        costs.append(float(ep_c))
        lens.append(int(info.get("episode_length", 0)))
        landed.append(bool(info.get("landed", False)))
    return rewards, costs, lens, landed


# ---------------------- EVAL WRAPPER ---------------------- #


def run_eval(
    name: str,
    noise_sigma: float,
    wind_level: str,
    enable_wind: bool,
    task: str,
    render: bool,
    policy,
    n_eps: int,
    *,  # keyword-only from here down
    fail_idx: int | None = None,
    fail_steps: int = 0,
    fail_delay_sec: float = 0.0,
) -> pd.DataFrame:
    env = make_env(task=task, render=render, enable_wind=enable_wind, wind_level=wind_level)

    if name.startswith("delay_"):
        steps = int(name.split("_", 1)[1])
        env.unwrapped.drone.args.delay_steps = steps
        env.unwrapped.drone.delay_buf = deque(maxlen=steps or 1)  # reset buffer

    if name.startswith("fail_"):
        env.unwrapped.drone.args.rotor_fail_idx = fail_idx
        env.unwrapped.drone.args.rotor_fail_steps = fail_steps

    rews, costs, lens, landed = rollout(
        policy=policy,
        env=env,
        n_eps=n_eps,
        noise_sigma=noise_sigma,
        fail_idx=fail_idx,
        fail_steps=fail_steps,
        fail_delay_sec=fail_delay_sec,
    )
    env.close()
    return pd.DataFrame(
        {
            "episode": np.arange(n_eps),
            f"reward_{name}": rews,
            f"cost_{name}": costs,
            f"len_{name}": lens,
            f"land_{name}": [int(x) for x in landed],
        }
    )


# ---------------------- POPULATE LIST VARIANTS ---------------------- #


def populateNoiseList(run_noise: bool, noise_sigmas: str):
    if run_noise:
        noise_list = [float(x.strip()) for x in noise_sigmas.split(",") if x.strip()]
        print(f"Noise List: {noise_list}")
        return noise_list
    return []


def populateWindList(run_wind: bool, wind_levels: str):
    if run_wind:
        wind_list = [x.strip() for x in wind_levels.split(",") if x.strip()]
        print(f"Wind List: {wind_list}")
        return wind_list
    return []


def populateDelayList(run_delay: bool, delay_steps: str):
    if run_delay:
        delay_list = [int(x.strip()) for x in delay_steps.split(",") if x.strip()]
        print(f"Delay List: {delay_list}")
        return delay_list
    return []


def populateRotorList(run_rotor_failure: bool, rotor_failures: str):
    rotor_list = []
    if run_rotor_failure:
        for item in rotor_failures.split(","):
            s = item.strip()
            if not s:
                continue
            # Expect format "idx:steps"
            parts = s.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid rotor_failure spec '{s}'. Expected 'idx:steps'.")
            idx = int(parts[0])
            steps = int(parts[1])
            if idx < 0 or idx > 3:
                raise ValueError(f"Rotor index must be 0..3, got {idx}.")
            if steps <= 0:
                raise ValueError(f"Rotor-failure steps must be >=1, got {steps}.")
            rotor_list.append((idx, steps))

    print(f"Rotor List: {rotor_list}")
    return rotor_list


# ---------------------- MAIN ---------------------- #
@pyrallis.wrap()
def main(cfg: EvalCfg = EvalCfg):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg, ckpt = exp_util.load_config_and_model(cfg.model_dir, best=True)
    task = train_cfg["task"]

    test_env = make_env(task, render=cfg.render)
    agent = PPOLagAgent(
        env=test_env,
        device="cpu",
        logger=None,
        **{
            k: train_cfg[k]
            for k in (
                "use_lagrangian",
                "thread",
                "seed",
                "cost_limit",
                "hidden_sizes",
                "unbounded",
                "last_layer_scale",
            )
        },
    )
    agent.policy.load_state_dict(ckpt["model"])
    agent.policy.eval()
    test_env.close()

    # --- Variant lists --------------------------------- #
    noise_list: List[float] = populateNoiseList(cfg.run_noise, cfg.noise_sigmas)
    wind_list: List[str] = populateWindList(cfg.run_wind, cfg.wind_levels)
    delay_list: List[int] = populateDelayList(cfg.run_delay, cfg.delay_steps)
    rotor_list = populateRotorList(cfg.run_rotor_fail, cfg.rotor_failures)

    eval_frames: Dict[str, pd.DataFrame] = {}

    # --- Clean baseline --------------------------------- #
    print("|CLEAN BASELINE|")
    eval_frames["clean"] = run_eval(
        name="clean",
        noise_sigma=0.0,
        wind_level="none",
        enable_wind=False,
        task=task,
        render=cfg.render,
        policy=agent.policy,
        n_eps=cfg.episodes,
    )

    # --- Gaussian noise variants --------------------------------- #
    for sigma in noise_list:
        print(f"|GAUSSIAN NOISE| Noise Sigma={sigma}")
        key = f"noise_{sigma:g}"
        eval_frames[key] = run_eval(
            name=key,
            noise_sigma=sigma,
            wind_level="none",
            enable_wind=False,
            task=task,
            render=cfg.render,
            policy=agent.policy,
            n_eps=cfg.episodes,
        )

    # --- Wind variants --------------------------------- #
    for level in wind_list:
        print(f"|WIND VARIANTS| Wind Level={level}")
        key = f"wind_{level}"
        enable = level.lower() != "none"
        eval_frames[key] = run_eval(
            name=key,
            noise_sigma=0.0,
            wind_level=level,
            enable_wind=enable,
            task=task,
            render=cfg.render,
            policy=agent.policy,
            n_eps=cfg.episodes,
        )

    # --- Delay variants --------------------------------- #
    for k in delay_list:
        print(f"|ACTION DELAY| Frame Delay={k}")
        key = f"delay_{k}"
        eval_frames[key] = run_eval(
            name=key,
            noise_sigma=0.0,
            wind_level="none",
            enable_wind=False,
            task=task,
            render=cfg.render,
            policy=agent.policy,
            n_eps=cfg.episodes,
        )

    # --- Rotor failure variants --------------------------------- #
    print("|ROTOR FAILURE|")
    if cfg.run_rotor_fail:
        FAIL_DELAY_SEC = 3.0

        for idx, steps in rotor_list:
            key = f"fail_{idx}_{steps}"
            eval_frames[key] = run_eval(
                name=key,
                noise_sigma=0.0,
                wind_level="none",
                enable_wind=False,
                task=task,
                render=cfg.render,
                policy=agent.policy,
                n_eps=cfg.episodes,
                fail_idx=idx,
                fail_steps=steps,
                fail_delay_sec=FAIL_DELAY_SEC,
            )

    # -------------------------- Combine & save per-episode summaries -------------------------- #
    summary = eval_frames["clean"].copy()
    for name, df_part in eval_frames.items():
        if name == "clean":
            continue
        summary = summary.merge(df_part, on="episode")
        summary[f"reward_diff_{name}"] = summary[f"reward_{name}"] - summary["reward_clean"]
        summary[f"cost_diff_{name}"] = summary[f"cost_{name}"] - summary["cost_clean"]

    numeric_cols = [c for c in summary.columns if summary[c].dtype.kind in "fi"]
    summary[numeric_cols] = summary[numeric_cols].apply(pd.to_numeric, downcast="float")

    summary_csv = out_dir / "episode_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Per-episode summary written to", summary_csv)

    # -------------------------- Landing success rates -------------------------- #
    success_rates = {format_name(k): df.filter(like="land_").iloc[:, 0].mean() for k, df in eval_frames.items()}

    pd.DataFrame([{"eval": k, "success_rate": v, "episodes": cfg.episodes} for k, v in success_rates.items()]).to_csv(
        out_dir / "landing_success_rates.csv", index=False
    )

    # -------------------------- Generate plots -------------------------- #
    for name in eval_frames:
        if name == "clean":
            continue
        plot_stacked_rewards(summary, "clean", name, out_dir / f"reward_stacked_clean_{name}.png")
        plot_reward_diff(summary, "clean", name, out_dir / f"reward_diff_{name}.png")
        plot_cost_box(summary, name, out_dir / f"cost_box_clean_{name}.png")

    plot_success_rates(success_rates, out_dir / "landing_success_rates.png")

    print("Saved all figures to", out_dir)


if __name__ == "__main__":
    main()
