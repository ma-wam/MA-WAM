"""Matched HCR-WM/monolithic candidate-ranking evaluation.

Both scorers receive exactly the same frozen policy, candidate budget, evaluation
seed, and receding-horizon harness. Only the loaded world-model scorer differs.
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
from world_model import MoEWorldModel
from world_model.baseline_world_model import BaselineWorldModel


def _atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".tmp_matched_scorer_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_worker_dimensions(worker, args):
    worker_agents = int(worker.Config.n_agents)
    normalizer_agents = int(worker.normalizer.n_agents)
    obs_dim = int(worker.normalizer.observation_dim)
    action_dim = int(worker.normalizer.action_dim)
    if worker_agents != normalizer_agents:
        raise ValueError(
            f"worker/normalizer n_agents mismatch: {worker_agents} != {normalizer_agents}"
        )
    if int(args.n_agents) != worker_agents:
        raise ValueError(
            f"CLI/worker n_agents mismatch: CLI={args.n_agents}, worker={worker_agents}"
        )
    if int(args.obs_dim) != obs_dim:
        raise ValueError(
            f"CLI/normalizer obs_dim mismatch: CLI={args.obs_dim}, normalizer={obs_dim}"
        )
    requested_action_dim = int(args.num_actions if args.discrete else args.act_dim)
    if requested_action_dim <= 0:
        raise ValueError("action dimension must be positive")
    if args.discrete:
        # SMAC data may store integer scalars or one-hot actions; its inverse
        # head and the world model always use num_actions logits.
        if action_dim not in (1, requested_action_dim):
            raise ValueError(
                "normalizer/CLI discrete action mismatch: "
                f"normalizer={action_dim}, num_actions={requested_action_dim}"
            )
    elif action_dim != requested_action_dim:
        raise ValueError(
            "normalizer/CLI action mismatch: "
            f"normalizer={action_dim}, act_dim={requested_action_dim}"
        )
    return worker_agents, obs_dim, requested_action_dim, action_dim


def _validate_checkpoint_shape(state_dict, scorer, n_agents, obs_dim, act_dim):
    if scorer == "moe":
        key = "dynamics.dispatch_weights"
        if key not in state_dict:
            raise ValueError(f"HCR-WM checkpoint missing {key}")
        # dispatch_weights has shape (slots, hidden_dim); it does not encode
        # obs_dim + act_dim. The model constructed from validated worker/CLI
        # dimensions is checked against every tensor by strict load below.
        return

    key = "dynamics.net.0.weight"
    if key not in state_dict:
        raise ValueError(f"monolithic checkpoint missing {key}")
    actual_input = int(state_dict[key].shape[1])
    expected_input = n_agents * (obs_dim + act_dim)
    if actual_input != expected_input:
        raise ValueError(
            f"{scorer} checkpoint/environment input mismatch: "
            f"checkpoint={actual_input}, expected={expected_input}"
        )


def _load_scorer(args, device, dimensions):
    n_agents, obs_dim, act_dim, _ = dimensions
    checkpoint_path = (
        args.wm if os.path.isabs(args.wm) else os.path.join(PROJECT_ROOT, args.wm)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    state_dict = checkpoint["model_state_dict"]
    _validate_checkpoint_shape(state_dict, args.scorer, n_agents, obs_dim, act_dim)
    if args.scorer == "moe":
        model = MoEWorldModel(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_dyn_experts=args.n_dyn,
            n_rew_experts=args.n_rew,
            hidden_dim=args.hidden,
            n_slots=args.n_slots,
        )
    else:
        checkpoint_type = checkpoint.get("model_type")
        if checkpoint_type not in (None, "monolithic"):
            raise ValueError(
                f"expected monolithic checkpoint, found model_type={checkpoint_type!r}"
            )
        checkpoint_hidden = checkpoint.get("hidden_dim")
        if checkpoint_hidden is not None and int(checkpoint_hidden) != int(args.hidden):
            raise ValueError(
                f"checkpoint/CLI hidden mismatch: {checkpoint_hidden} != {args.hidden}"
            )
        model = BaselineWorldModel(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.hidden,
            model_type="monolithic",
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), os.path.abspath(checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wm", required=True)
    parser.add_argument("--scorer", choices=["moe", "monolithic"], required=True)
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
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_dyn", type=int, default=8)
    parser.add_argument("--n_rew", type=int, default=4)
    parser.add_argument("--n_slots", type=int, default=4)
    parser.add_argument("--save_ep", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_dir = (
        args.run_dir
        if os.path.isabs(args.run_dir)
        else os.path.join(PROJECT_ROOT, args.run_dir)
    )
    worker = plan_eval.load_worker(run_dir)
    device = worker.Config.device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    load_step = (
        args.load_step if args.load_step > 0 else plan_eval.find_best_step(run_dir)
    )
    plan_eval.load_policy_ckpt(worker, load_step)
    if args.test_ret >= 0:
        print(f"Override test_ret: {worker.Config.test_ret} -> {args.test_ret}")
        worker.Config.test_ret = args.test_ret
    worker.Config.num_eval = args.num_eval
    worker.Config.num_envs = min(args.num_eval, worker.Config.num_envs)
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    dimensions = _validate_worker_dimensions(worker, args)
    scorer, checkpoint_path = _load_scorer(args, device, dimensions)
    requested_horizon = int(args.H)
    effective_horizon = min(requested_horizon, int(worker.Config.horizon) - 1)
    if effective_horizon <= 0:
        raise ValueError(
            f"policy horizon={worker.Config.horizon} yields no executable actions"
        )
    base_seed = int(getattr(worker.Config, "seed", 0))
    num_envs = int(worker.Config.num_envs)
    n_params = sum(parameter.numel() for parameter in scorer.parameters())
    print(f"=== run_dir={run_dir}")
    print(f"=== scorer={args.scorer} wm={checkpoint_path} params={n_params:,}")
    print(
        "=== dimensions="
        f"n_agents={dimensions[0]} obs_dim={dimensions[1]} "
        f"wm_action_dim={dimensions[2]} normalizer_action_dim={dimensions[3]}"
    )
    print(
        f"=== M={args.M} requested_H={requested_horizon} "
        f"effective_H={effective_horizon} steps={args.steps} num_eval={args.num_eval}"
    )

    plan_eval.rebuild_envs_for_arm(worker, base_seed)
    plan_eval.set_seed(base_seed)
    print(
        f"[PAIRED_ENV] arm={args.scorer} base_seed={base_seed} "
        f"num_envs={worker.Config.num_envs}"
    )
    all_rewards, all_wins, all_episode_rewards = [], [], []
    remaining = args.num_eval
    while remaining > 0:
        batch_size = min(remaining, num_envs)
        rewards, wins, episode_rewards = plan_eval.planning_eval(
            worker,
            scorer,
            batch_size,
            args.M,
            requested_horizon,
            device,
            num_actions=args.num_actions,
            discrete=args.discrete,
            select="wm",
        )
        all_rewards.extend(rewards.tolist())
        all_wins.extend(wins.tolist())
        all_episode_rewards.extend(float(reward.mean()) for reward in episode_rewards)
        remaining -= batch_size

    tag = f"{args.scorer}-sel(M={args.M})"
    print(
        f"[RESULT] {tag} mean_ep_reward={np.mean(all_rewards):.3f} "
        f"win_rate={np.mean(all_wins):.3f} std={np.std(all_episode_rewards):.3f} "
        f"(n={len(all_episode_rewards)})"
    )
    output = {
        "tag": tag,
        "rews": all_episode_rewards,
        "wins": [float(value) for value in all_wins],
        "scorer": args.scorer,
        "wm": checkpoint_path,
        "M": int(args.M),
        "H": requested_horizon,
        "requested_H": requested_horizon,
        "effective_H": effective_horizon,
        "load_step": int(load_step),
        "paired_env_seed": base_seed,
    }
    output_path = (
        args.save_ep if args.save_ep.endswith(".json") else f"{args.save_ep}_{tag}.json"
    )
    _atomic_json_dump(output, output_path)
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
