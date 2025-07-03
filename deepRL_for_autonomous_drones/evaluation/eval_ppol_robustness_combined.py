import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
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
    plot_action_profiles,
    plot_combined_metric,
    plot_combined_success,
    plot_failure_causes,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="is_categorical_dtype is deprecated and will be removed",
)


# -------------------------- CONFIGS -------------------------- #
@dataclass
class EvalCfg:
    # model_dir: str = "../training/benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-6/ppol-no-wind-23"
    model_dir: str = "../training/benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-6/ppol-10"
    episodes: int = 20
    render: bool = False
    out_dir: str = "evaluation/action_noise_combination"

    noise_sigmas: str = "0.05, 0.1, 0.5, 1.0"  # e.g. "0.05,0.1,0.5,1.0" or "none"
    wind_levels: str = "light_breeze, light, medium, high"  # light_breeze,light,medium,high
    delay_steps: str = "1, 2, 4, 5"
    rotor_failures: str = "0:5, 1:5"  # comma-separated "idx:steps", e.g. "0:10,2:20", or "none"
    sign_flips: str = "0;1"  # example: "0,2" -> test rotor 0 and rotor 2 sign flip together
    #                          "0,2;1,3" -> test r0, r2 together, then test r1, r3 together

    run_noise: bool = True
    run_wind: bool = True
    run_delay: bool = True

    def __post_init__(self):
        self.run_noise = self.noise_sigmas.strip().lower() != "none"
        self.run_wind = self.wind_levels.strip().lower() != "none"
        self.run_delay = self.delay_steps.strip().lower() != "none"
        self.run_rotor_fail = self.rotor_failures.strip().lower() != "none"
        self.run_flip = self.sign_flips.strip().lower() != "none"


# -------------------------- ENVS -------------------------- #


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


# -------------------------- HELPERS -------------------------- #


def to_batch(obs):
    if isinstance(obs, dict):
        obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    else:
        obs = np.expand_dims(obs, 0)
    return Batch(obs=obs)


# -------------------------- ROLLOUT -------------------------- #


def rollout(
    policy,
    env: gym.Env,
    n_eps: int,
    noise_sigma: float = 0.0,
    *,
    fail_idx: int | None = None,
    fail_steps: int = 0,
    fail_delay_sec: float = 0.0,
):
    """
    Returns:
        rewards, costs, lens, landed, infos
        infos is a list of dicts (one per episode) that already
        contains 'failure' and 'act_mean_0..3'.
    """
    drone = env.unwrapped.drone
    ctrl_freq = env.unwrapped.CTRL_FREQ

    # noise toggle
    drone.args.noise_sigma = noise_sigma
    drone.args.enabled = noise_sigma > 0

    rewards, costs, lens, landed = [], [], [], []
    infos: list[dict] = []

    for _ in range(n_eps):
        # schedule rotor failure for this episode
        if fail_idx is not None and fail_steps > 0:
            drone.armRotorFailure(
                idx=fail_idx,
                steps=fail_steps,
                delay_sec=fail_delay_sec,
                ctrl_freq=ctrl_freq,
            )
        else:
            drone.resetRotorFailure()

        seed = random.randint(0, 10_000)
        obs, _ = env.reset(seed=seed)

        done = trunc = False
        ep_r = ep_c = 0.0
        steps_in_ep = 0
        action_sum = np.zeros(4, dtype=np.float64)

        while not (done or trunc):
            with torch.no_grad():
                act = policy(to_batch(obs), deterministic=True).act.squeeze(0)

            action_sum += np.asarray(act, dtype=np.float64)
            steps_in_ep += 1

            obs, r, done, trunc, info = env.step(act)
            ep_r += r
            ep_c += info.get("cost", 0.0)

        if info.get("landed", False):
            failure_lbl = "SUCCESS"
        else:
            p = env.unwrapped.getBulletClient()
            drone_id = env.unwrapped.drone.getDroneID()
            plane_id = env.unwrapped.plane

            plane_hits = p.getContactPoints(bodyA=drone_id, bodyB=plane_id)
            tree_hits = any(p.getContactPoints(bodyA=drone_id, bodyB=t) for t in env.unwrapped.trees)

            if plane_hits:
                failure_lbl = "Plane crash"
            elif tree_hits:
                failure_lbl = "Tree crash"
            elif trunc:
                failure_lbl = "Timeout"
            elif abs(env.drone.get_position()[2]) > env.MAX_Z:
                failure_lbl = "Out-of-bounds"
            else:
                vz = env.drone.get_linear_velocity()[2]
                failure_lbl = "Hard landing" if abs(vz) > 0.5 else "Other"
        # -------------------------------------------------------------

        act_mean = action_sum / max(1, steps_in_ep)

        info_ep = {
            "failure": failure_lbl,
            "act_mean_0": act_mean[0],
            "act_mean_1": act_mean[1],
            "act_mean_2": act_mean[2],
            "act_mean_3": act_mean[3],
            "episode_length": steps_in_ep,
        }
        infos.append(info_ep)

        rewards.append(float(ep_r))
        costs.append(float(ep_c))
        lens.append(int(steps_in_ep))
        landed.append(bool(info.get("landed", False)))

    return rewards, costs, lens, landed, infos


# -------------------------- EVAL WRAPPER -------------------------- #


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
    delay_steps: int = 0,
    fail_idx: int | None = None,
    fail_steps: int = 0,
    fail_delay_sec: float = 0.0,
    flip_idxs: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    env = make_env(task=task, render=render, enable_wind=enable_wind, wind_level=wind_level)

    # ------------- RESET GLOBAL ROBUSTNESS FLAGS -------------- #
    drone = env.unwrapped.drone
    drone_args = drone.args
    drone_args.delay_steps = 0
    drone_args.noise_sigma = 0.0
    drone_args.enabled = False
    drone_args.rotor_fail_idx = None
    drone_args.rotor_fail_steps = 0
    drone.setSignFlips(set())
    # ---------------------------------------------------------- #

    if delay_steps > 0:
        env.unwrapped.drone.args.delay_steps = delay_steps
        env.unwrapped.drone.delay_buf = deque(maxlen=delay_steps)

    if name.startswith("fail_"):
        env.unwrapped.drone.args.rotor_fail_idx = fail_idx
        env.unwrapped.drone.args.rotor_fail_steps = fail_steps

    if flip_idxs:
        drone.setSignFlips(set(flip_idxs))

    if noise_sigma > 0.0:
        drone.args.noise_sigma = noise_sigma

    # Enable robustness if any of the settings are set
    any_robust = delay_steps > 0 or noise_sigma > 0.0 or (fail_idx is not None and fail_steps > 0) or (flip_idxs and len(flip_idxs) > 0)
    drone_args.enabled = any_robust

    rews, costs, lens, landed, infos = rollout(
        policy=policy,
        env=env,
        n_eps=n_eps,
        noise_sigma=noise_sigma,
        fail_idx=fail_idx,
        fail_steps=fail_steps,
        fail_delay_sec=fail_delay_sec,
    )

    env.close()

    df = pd.DataFrame(
        {
            "episode": np.arange(n_eps),
            f"reward_{name}": rews,
            f"cost_{name}": costs,
            f"len_{name}": lens,
            f"land_{name}": [int(x) for x in landed],
            "failure": [d["failure"] for d in infos],
            "act_m0": [d["act_mean_0"] for d in infos],
            "act_m1": [d["act_mean_1"] for d in infos],
            "act_m2": [d["act_mean_2"] for d in infos],
            "act_m3": [d["act_mean_3"] for d in infos],
        }
    )

    return df


# -------------------------- POPULATE LIST VARIANTS -------------------------- #


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


def populateSignFlipList(run_flip: bool, sign_flips: str) -> List[Tuple[int, ...]]:
    if (not run_flip) or sign_flips.strip().lower() == "none":
        return []

    variants: List[Tuple[int, ...]] = []
    for group in sign_flips.split(";"):
        group = group.strip()
        if not group:
            continue
        idxs = tuple(sorted({int(i.strip()) for i in group.split(",") if i.strip()}))
        if any(i < 0 or i > 3 for i in idxs):
            raise ValueError(f"Rotor index must be 0‥3 in --sign_flips, got {idxs}")
        variants.append(idxs)

    return variants


# -------------------------- MAIN -------------------------- #
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
    flip_list: List[Tuple[int, ...]] = populateSignFlipList(cfg.run_flip, cfg.sign_flips)

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

    combinations: list[tuple[float | None, str | None, int | None, tuple[int, int] | None, Tuple[int, ...] | None]] = []
    for noise in [None] + noise_list or [None]:
        for wind in [None] + wind_list or [None]:
            for delay in [None] + delay_list or [None]:
                for rotor in [None] + rotor_list or [None]:
                    for flip in ([None] + flip_list) or [None]:
                        if all(v is None for v in (noise, wind, delay, rotor, flip)):
                            continue  # skip the clean case
                        combinations.append((noise, wind, delay, rotor, flip))
    print(f"Total combinations: {len(combinations)}")
    FAIL_DELAY_SEC = 3.0

    for noise, wind, delay, rotor, flip in combinations:
        parts = []
        if noise is not None:
            parts.append(f"noise_{noise:g}")
        if delay is not None:
            parts.append(f"delay_{delay}")
        if wind is not None:
            parts.append(f"wind_{wind}")
        if rotor is not None:
            idx, steps = rotor
            parts.append(f"fail_{idx}_{steps}")
        if flip is not None:
            parts.append("flip_" + ",".join(str(i) for i in flip))
        key = "+".join(parts)

        print(f"|COMBO| {key}")

        eval_frames[key] = run_eval(
            name=key,
            noise_sigma=noise or 0.0,
            wind_level=wind or "none",
            enable_wind=bool(wind and wind != "none"),
            task=task,
            render=cfg.render,
            policy=agent.policy,
            n_eps=cfg.episodes,
            delay_steps=delay or 0,
            fail_idx=rotor[0] if rotor else None,
            fail_steps=rotor[1] if rotor else 0,
            fail_delay_sec=FAIL_DELAY_SEC if rotor else 0.0,
            flip_idxs=flip,
        )

    all_failures: list[str] = []
    for df in eval_frames.values():
        if "failure" in df.columns:
            all_failures.extend(df["failure"].tolist())

    # --------------------------------- Combine & save per-episode summaries --------------------------------- #
    summary = eval_frames["clean"].copy()
    for name, df_part in eval_frames.items():
        if name == "clean":
            continue

        # --- drop columns that would clash with summary ----------- #
        df_part = df_part.drop(
            columns=["failure", "act_m0", "act_m1", "act_m2", "act_m3"],
            errors="ignore",
        )
        # ------------------------------------------------------------ #

        summary = summary.merge(df_part, on="episode")

        summary[f"reward_diff_{name}"] = summary[f"reward_{name}"] - summary["reward_clean"]
        summary[f"cost_diff_{name}"] = summary[f"cost_{name}"] - summary["cost_clean"]

    numeric_cols = [c for c in summary.columns if summary[c].dtype.kind in "fi"]
    summary[numeric_cols] = summary[numeric_cols].apply(pd.to_numeric, downcast="float")

    summary_csv = out_dir / "episode_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Per-episode summary written to", summary_csv)

    # ------------------------- Landing success rates ------------------------- #
    success_raw: Dict[str, float] = {k: df.filter(like="land_").iloc[:, 0].mean() for k, df in eval_frames.items()}
    success_rates = {format_name(k): v for k, v in success_raw.items()}

    pd.DataFrame([{"eval": format_name(k), "success_rate": v, "episodes": cfg.episodes} for k, v in success_raw.items()]).to_csv(
        out_dir / "landing_success_rates.csv", index=False
    )

    variant_keys = [k for k in success_raw if k != "clean" and success_raw[k] > 0.0]
    success_rates = {format_name(k): r for k, r in success_raw.items() if r > 0.0}

    plot_combined_metric(summary, variant_keys, "reward_", out_dir / "combined_reward_boxplot.png")
    plot_combined_metric(summary, variant_keys, "cost_", out_dir / "combined_cost_boxplot.png")
    plot_combined_success(success_rates, out_dir / "combined_landing_success_rates.png")
    plot_failure_causes(all_failures, out_dir / "failure_causes.png")
    plot_action_profiles(summary, out_dir / "action_profile_comparison.png")

    print("✓ All evaluations done — results saved to", out_dir)


if __name__ == "__main__":
    main()
