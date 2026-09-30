"""Evaluate a centralized Q ranker on exactly the same CoFlow candidates.

This is a matched selector control for ``plan_eval_scorer.py``.  Candidate
generation, inverse dynamics, legal-action masking, receding-horizon execution,
and evaluation seeds all reuse ``run_scripts.plan_eval``.  Only the scoring
function changes from an H-step world-model rollout to Q(o_t, a_t).
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import run_scripts.plan_eval_paired as plan_eval
from world_model.q_ranker import load_fqe_critic, q_rank_score


def _atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_selector_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_dimensions(config, args, worker):
    """Derive environment dimensions from the loaded worker, then cross-check all inputs."""

    worker_agents = int(worker.Config.n_agents)
    normalizer_agents = int(worker.normalizer.n_agents)
    worker_obs_dim = int(worker.normalizer.observation_dim)
    if worker_agents != normalizer_agents:
        raise ValueError(
            f"worker/normalizer n_agents mismatch: {worker_agents} != {normalizer_agents}"
        )
    if args.n_agents != worker_agents:
        raise ValueError(
            f"CLI/worker n_agents mismatch: CLI={args.n_agents}, worker={worker_agents}"
        )
    if args.obs_dim != worker_obs_dim:
        raise ValueError(
            f"CLI/normalizer obs_dim mismatch: CLI={args.obs_dim}, normalizer={worker_obs_dim}"
        )

    expected_obs = worker_agents * worker_obs_dim
    critic_act_dim = int(config["act_dim"])
    if critic_act_dim % worker_agents:
        raise ValueError(
            f"critic joint act_dim={critic_act_dim} is not divisible by n_agents={worker_agents}"
        )
    critic_per_agent_act = critic_act_dim // worker_agents
    if args.discrete:
        if args.num_actions <= 0:
            raise ValueError("--num_actions must be positive for a discrete Q ranker")
        if args.num_actions != critic_per_agent_act:
            raise ValueError(
                "CLI/critic discrete action mismatch: "
                f"num_actions={args.num_actions}, critic_per_agent={critic_per_agent_act}"
            )
        expected_act = worker_agents * args.num_actions
    else:
        normalizer_act_dim = int(worker.normalizer.action_dim)
        if args.act_dim != normalizer_act_dim:
            raise ValueError(
                "CLI/normalizer action mismatch: "
                f"act_dim={args.act_dim}, normalizer={normalizer_act_dim}"
            )
        if normalizer_act_dim != critic_per_agent_act:
            raise ValueError(
                "normalizer/critic action mismatch: "
                f"normalizer={normalizer_act_dim}, critic_per_agent={critic_per_agent_act}"
            )
        expected_act = worker_agents * normalizer_act_dim

    actual_obs = int(config["obs_dim"])
    actual_act = critic_act_dim
    if actual_obs != expected_obs or actual_act != expected_act:
        raise ValueError(
            "FQE/environment dimension mismatch: "
            f"checkpoint=(obs={actual_obs}, act={actual_act}), "
            f"evaluation=(obs={expected_obs}, act={expected_act})"
        )
    if "n_agents" in config and int(config["n_agents"]) != worker_agents:
        raise ValueError(
            f"checkpoint/worker n_agents mismatch: {config['n_agents']} != {worker_agents}"
        )
    return worker_agents, worker_obs_dim, critic_per_agent_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--n_agents", type=int, required=True)
    parser.add_argument("--obs_dim", type=int, required=True)
    parser.add_argument("--num_actions", type=int, default=0)
    parser.add_argument("--act_dim", type=int, default=0)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--test_ret", type=float, default=-1.0)
    parser.add_argument("--save_ep", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    critic_path = args.critic if os.path.isabs(args.critic) else os.path.join(PROJECT_ROOT, args.critic)

    worker = plan_eval.load_worker(run_dir)
    device = worker.Config.device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    load_step = args.load_step if args.load_step > 0 else plan_eval.find_best_step(run_dir)
    plan_eval.load_policy_ckpt(worker, load_step)
    if args.test_ret >= 0:
        print(f"Override test_ret: {worker.Config.test_ret} -> {args.test_ret}")
        worker.Config.test_ret = args.test_ret
    worker.Config.num_eval = args.num_eval
    worker.Config.num_envs = min(args.num_eval, worker.Config.num_envs)
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    critic, critic_config, critic_path = load_fqe_critic(critic_path, device)
    derived_dims = _validate_dimensions(critic_config, args, worker)
    requested_horizon = int(args.H)
    effective_horizon = min(requested_horizon, int(worker.Config.horizon) - 1)
    if effective_horizon <= 0:
        raise ValueError(
            f"policy horizon={worker.Config.horizon} yields no executable candidate actions"
        )
    n_params = sum(parameter.numel() for parameter in critic.parameters())
    print(f"=== run_dir={run_dir}")
    print(f"=== q_ranker={critic_path} params={n_params:,}")
    print(
        "=== worker_dims="
        f"n_agents={derived_dims[0]} obs_dim={derived_dims[1]} "
        f"critic_action_dim_per_agent={derived_dims[2]}"
    )
    print(
        f"=== selector=Q(o_t,a_t) steps={args.steps} "
        f"requested_H={requested_horizon} effective_H={effective_horizon} "
        f"M={args.M} num_eval={args.num_eval}"
    )
    if requested_horizon != effective_horizon:
        print(
            f"[HORIZON] requested_H={requested_horizon}; policy candidates contain "
            f"effective_H={effective_horizon} executable actions"
        )

    # planning_eval resolves wm_score in its defining module.  Rebinding it in
    # this isolated executable preserves the established candidate/eval harness
    # without modifying existing world-model experiments.
    original_score = plan_eval.wm_score
    plan_eval.wm_score = q_rank_score
    try:
        base_seed = getattr(worker.Config, "seed", 0)
        num_envs = worker.Config.num_envs
        plan_eval.rebuild_envs_for_arm(worker, base_seed)
        plan_eval.set_seed(base_seed)
        print(
            f"[PAIRED_ENV] arm=q-rank(M={args.M}) base_seed={base_seed} "
            f"num_envs={worker.Config.num_envs}"
        )
        all_rewards, all_wins, all_episode_rewards = [], [], []
        remaining = args.num_eval
        while remaining > 0:
            n_envs = min(remaining, num_envs)
            rewards, wins, episode_rewards = plan_eval.planning_eval(
                worker,
                critic,
                n_envs,
                args.M,
                args.H,
                device,
                num_actions=args.num_actions,
                discrete=args.discrete,
                select="wm",
            )
            all_rewards.extend(rewards.tolist())
            all_wins.extend(wins.tolist())
            all_episode_rewards.extend([reward.mean() for reward in episode_rewards])
            remaining -= n_envs
    finally:
        plan_eval.wm_score = original_score

    tag = f"q-rank(M={args.M})"
    print(
        f"[RESULT] {tag}  mean_ep_reward={np.mean(all_rewards):.3f}  "
        f"win_rate={np.mean(all_wins):.3f}  std={np.std(all_episode_rewards):.3f}  "
        f"p10={np.percentile(all_episode_rewards, 10):.3f}  (n={len(all_rewards)})"
    )
    if args.save_ep:
        output = {
            "tag": tag,
            "rews": [float(value) for value in all_episode_rewards],
            "wins": [float(value) for value in all_wins],
            "scorer": "centralized_behavior_fqe",
            "critic": critic_path,
            "M": args.M,
            "H": requested_horizon,
            "requested_H": requested_horizon,
            "effective_H": effective_horizon,
            "load_step": load_step,
            "paired_env_seed": int(base_seed),
        }
        output_path = args.save_ep if args.save_ep.endswith(".json") else f"{args.save_ep}_{tag}.json"
        _atomic_json_dump(output, output_path)
        print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
