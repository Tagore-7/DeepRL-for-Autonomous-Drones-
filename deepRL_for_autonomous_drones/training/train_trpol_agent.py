import os
import types
from dataclasses import asdict
import pandas as pd

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pyrallis
from tianshou.env import BaseVectorEnv, DummyVectorEnv, RayVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from fsrl.agent import TRPOLagAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name
from fsrl.data import FastCollector
from deepRL_for_autonomous_drones.config.trpol_cfg import TrainCfg, DroneLandingCfg
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController

TASK_TO_CFG = {"SafetyDroneLanding-v0": DroneLandingCfg}
WORKER_MAPPING = {
    "BaseVectorEnv": BaseVectorEnv,
    "DummyVectorEnv": DummyVectorEnv,
    "RayVectorEnv": RayVectorEnv,
    "SubprocVectorEnv": SubprocVectorEnv,
    "ShmemVectorEnv": ShmemVectorEnv,
}


def make_training_env(task):
    env = gym.make(task)
    env = FlattenObservation(env)
    return env


@pyrallis.wrap()
def train(args: TrainCfg = TrainCfg):

    # update config
    cfg, old_cfg = asdict(args), asdict(TrainCfg())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(TASK_TO_CFG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(TASK_TO_CFG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(args.cost_limit)
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # demo_env = gym.make(args.task)
    demo_env = make_training_env(args.task)

    agent = TRPOLagAgent(
        env=demo_env,
        logger=logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = WORKER_MAPPING.get(args.worker)
    if worker is None:
        raise ValueError(f"Unknown worker type: {args.worker}")
    # train_envs = worker([lambda: gym.make(args.task) for _ in range(training_num)])
    # test_envs = worker([lambda: gym.make(args.task) for _ in range(args.testing_num)])

    train_envs = worker([make_training_env(args.task) for _ in range(training_num)])
    test_envs = worker([make_training_env(args.task) for _ in range(args.testing_num)])

    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        buffer_size=args.buffer_size,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
    )

    if __name__ == "__main__":
        # env = gym.make(args.task)
        env = make_training_env(args.task)

        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        final_eval_reward = rews.mean().item()
        final_eval_cost = cost.mean().item()
        csv_path = os.path.join(args.logdir, args.name, "progress.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "task": args.task,
                    "algo": args.prefix,
                    "reward": final_eval_reward,
                    "cost": final_eval_cost,
                }
            ]
        )
        df.to_csv(csv_path, index=False)
        print(f"Final eval CSV saved to {csv_path}")

        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
