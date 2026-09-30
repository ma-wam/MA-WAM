"""Evaluate direct trajectory-return ranking with the existing CoFlow proposals.

This script deliberately reuses ``planning_eval`` unchanged.  It only replaces
the score function: all policy candidates, inverse-dynamics decoding, legal-action
handling, receding-horizon execution, seeds, and environment logic remain the
same as in the HCR-WM/monolithic-scorer evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import run_scripts.plan_eval_paired as plan_eval_module
from world_model.trajectory_return_predictor import TrajectoryReturnPredictor


def atomic_json_dump(payload: Dict, path: str) -> None:
    """Write a complete JSON result before atomically publishing it."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_selector_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_predictor(path: str, device: torch.device) -> Tuple[TrajectoryReturnPredictor, Dict]:
    checkpoint = torch.load(path, map_location=device)
    if "model_config" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_config: {path}")
    model = TrajectoryReturnPredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def trajectory_return_score(
    predictor: TrajectoryReturnPredictor,
    start_obs_raw: np.ndarray,
    act_seqs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Score ``(M,E,H,A,ad)`` proposals without predicting future states."""
    M, E, H, A, action_dim = act_seqs.shape
    if H != predictor.horizon:
        raise ValueError(
            f"Candidate effective H={H} differs from predictor H={predictor.horizon}. "
            "Train the predictor at the policy's effective action horizon."
        )
    if (A, action_dim) != (predictor.n_agents, predictor.act_dim):
        raise ValueError(
            "Candidate action shape mismatch: "
            f"got A={A}, act_dim={action_dim}; model expects "
            f"A={predictor.n_agents}, act_dim={predictor.act_dim}"
        )
    expected_obs = (E, predictor.n_agents, predictor.obs_dim)
    if tuple(start_obs_raw.shape) != expected_obs:
        raise ValueError(
            f"Current observation mismatch: expected {expected_obs}, got {start_obs_raw.shape}"
        )
    observations = torch.as_tensor(
        start_obs_raw, dtype=torch.float32, device=device
    )
    observations = (
        observations.unsqueeze(0)
        .expand(M, E, A, predictor.obs_dim)
        .reshape(M * E, A, predictor.obs_dim)
        .contiguous()
    )
    actions = torch.as_tensor(act_seqs, dtype=torch.float32, device=device)
    actions = actions.reshape(M * E, H, A, action_dim)
    return predictor(observations, actions).reshape(M, E)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument(
        "--H", type=int, default=0, help="0 uses the predictor training horizon"
    )
    parser.add_argument("--num_eval", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--test_ret", type=float, default=-1.0)
    parser.add_argument(
        "--ablate_random",
        action="store_true",
        help="also evaluate random selection with paired seeds/candidate generation",
    )
    parser.add_argument("--save_ep", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_dir = (
        args.run_dir
        if os.path.isabs(args.run_dir)
        else os.path.join(PROJECT_ROOT, args.run_dir)
    )
    predictor_path = (
        args.predictor
        if os.path.isabs(args.predictor)
        else os.path.join(PROJECT_ROOT, args.predictor)
    )
    worker = plan_eval_module.load_worker(run_dir)
    device = worker.Config.device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    predictor, checkpoint = load_predictor(predictor_path, device)
    data_config = checkpoint.get("data_config", {})
    discrete = bool(data_config.get("discrete_action", False))
    num_actions = int(data_config.get("num_actions") or 0)
    requested_horizon = int(args.H or predictor.horizon)
    effective_horizon = int(predictor.horizon)
    if requested_horizon < effective_horizon:
        raise ValueError(
            f"requested_H={requested_horizon} is shorter than predictor/effective_H="
            f"{effective_horizon}"
        )

    if int(worker.Config.n_agents) != predictor.n_agents:
        raise ValueError(
            f"Policy has {worker.Config.n_agents} agents, predictor has {predictor.n_agents}"
        )
    obs_dim = int(worker.normalizer.observation_dim)
    if obs_dim != predictor.obs_dim:
        raise ValueError(
            f"Policy observation_dim={obs_dim}, predictor obs_dim={predictor.obs_dim}"
        )
    if discrete and num_actions != predictor.act_dim:
        raise ValueError(
            f"Checkpoint num_actions={num_actions}, predictor act_dim={predictor.act_dim}"
        )

    load_step = (
        args.load_step
        if args.load_step > 0
        else plan_eval_module.find_best_step(run_dir)
    )
    plan_eval_module.load_policy_ckpt(worker, load_step)
    if args.test_ret >= 0:
        print(f"Override test_ret: {worker.Config.test_ret} -> {args.test_ret}")
        worker.Config.test_ret = args.test_ret
    worker.Config.num_eval = args.num_eval
    worker.Config.num_envs = min(args.num_eval, worker.Config.num_envs)
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    # planning_eval resolves this name from its own module globals.
    plan_eval_module.wm_score = trajectory_return_score
    print(f"=== run_dir={run_dir}")
    print(
        f"=== direct-return predictor={predictor_path} "
        f"params={sum(p.numel() for p in predictor.parameters()):,}"
    )
    print(
        f"=== training_step={checkpoint.get('step')} policy_step={load_step} "
        f"M={args.M} requested_H={requested_horizon} "
        f"effective_H={effective_horizon} denoise_steps={args.steps} "
        f"num_eval={args.num_eval}"
    )
    if requested_horizon != effective_horizon:
        print(
            f"[HORIZON] requested_H={requested_horizon}; policy candidates are "
            f"scored at effective_H={effective_horizon} (predictor training horizon)"
        )

    arms = [("direct-return", "wm")]
    if args.ablate_random:
        arms.insert(0, ("random", "random"))
    base_seed = getattr(worker.Config, "seed", 0)
    num_envs = worker.Config.num_envs
    for arm_name, selection in arms:
        plan_eval_module.rebuild_envs_for_arm(worker, base_seed)
        plan_eval_module.set_seed(base_seed)
        print(
            f"[PAIRED_ENV] arm={arm_name} base_seed={base_seed} "
            f"num_envs={worker.Config.num_envs}"
        )
        rewards, wins, episode_rewards = [], [], []
        remaining = args.num_eval
        while remaining > 0:
            current = min(remaining, num_envs)
            reward_array, win_array, reward_vectors = plan_eval_module.planning_eval(
                worker,
                predictor,
                current,
                args.M,
                requested_horizon,
                device,
                num_actions=num_actions,
                discrete=discrete,
                select=selection,
            )
            rewards.extend(reward_array.tolist())
            wins.extend(win_array.tolist())
            episode_rewards.extend([float(x.mean()) for x in reward_vectors])
            remaining -= current

        tag = f"{arm_name}-sel(M={args.M})"
        result = {
            "tag": tag,
            "rews": episode_rewards,
            "wins": wins,
            "mean_ep_reward": float(np.mean(rewards)),
            "std_ep_reward": float(np.std(episode_rewards)),
            "win_rate": float(np.mean(wins)),
            "predictor": predictor_path,
            "M": args.M,
            "H": requested_horizon,
            "requested_H": requested_horizon,
            "effective_H": effective_horizon,
            "load_step": load_step,
            "paired_env_seed": int(base_seed),
            "predictor_step": checkpoint.get("step"),
        }
        print(
            f"[RESULT] {tag} mean_ep_reward={result['mean_ep_reward']:.3f} "
            f"win_rate={result['win_rate']:.3f} std={result['std_ep_reward']:.3f} "
            f"p10={np.percentile(episode_rewards, 10):.3f} (n={len(rewards)})"
        )
        if args.save_ep:
            if args.save_ep.endswith(".json") and len(arms) == 1:
                output_path = args.save_ep
            else:
                output_path = f"{args.save_ep}_{tag}.json"
            output_path = (
                output_path
                if os.path.isabs(output_path)
                else os.path.join(PROJECT_ROOT, output_path)
            )
            atomic_json_dump(result, output_path)
            print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
