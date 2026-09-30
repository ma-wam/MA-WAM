#!/usr/bin/env python3
"""Controlled same-state scorer and cross-agent influence evaluation.

All scorers receive the same policy-generated candidate set at every anchor
state. MPE and MA-MuJoCo are restored exactly before each real rollout, so
differences are attributable to the scorer rather than candidate sampling.
"""

import argparse
import csv
import json
import os
import sys
from collections import deque

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_scripts.eval_same_state_ranking import (  # noqa: E402
    corrcoef,
    generate_candidates,
    pairwise_acc,
    rankdata,
    restore_mpe_env,
    restore_mujoco_env,
    rollout_candidate,
    snapshot_mpe_env,
    snapshot_mujoco_env,
)
from run_scripts.plan_eval import (  # noqa: E402
    find_best_step,
    load_policy_ckpt,
    load_worker,
    load_wm,
    set_seed,
    to_np,
    wm_score,
)
from run_scripts.profile_plan_timing_all import infer_dims, run_dir_for, wm_path  # noqa: E402
from world_model.baseline_world_model import BaselineWorldModel  # noqa: E402


SCORERS = ("hcr", "monolithic", "independent")


def load_baseline(kind, checkpoint, n_agents, obs_dim, act_dim, device, hidden=512):
    model = BaselineWorldModel(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden,
        model_type=kind,
    )
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def one_step_from_snapshot(env, action, snapshot, restore_fn):
    restore_fn(env, snapshot)
    next_obs, reward, done, info = env.step(np.asarray(action))
    next_obs = np.asarray(next_obs, dtype=np.float32)
    restore_fn(env, snapshot)
    return next_obs


def choose_alternative(first_actions, source):
    base = first_actions[0, source]
    distances = np.linalg.norm(first_actions[:, source] - base, axis=-1)
    best = int(np.argmax(distances))
    return best, float(distances[best])


@torch.no_grad()
def model_influence(model, raw_obs, first_actions, device):
    """Return target-by-source one-step influence magnitudes."""
    n_agents = raw_obs.shape[1]
    base = first_actions[0].copy()
    action_batch = [base]
    for source in range(n_agents):
        alt_idx, distance = choose_alternative(first_actions, source)
        perturbed = base.copy()
        if distance <= 1e-8:
            perturbed[source] = np.clip(perturbed[source] + 0.05, -1.0, 1.0)
        else:
            perturbed[source] = first_actions[alt_idx, source]
        action_batch.append(perturbed)

    obs_batch = np.repeat(raw_obs, n_agents + 1, axis=0)
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.asarray(action_batch), dtype=torch.float32, device=device)
    pred = to_np(model.predict_next_obs(obs_t, act_t))
    influence = np.zeros((n_agents, n_agents), dtype=np.float64)
    for source in range(n_agents):
        delta = pred[source + 1] - pred[0]
        influence[:, source] = np.linalg.norm(delta, axis=-1)
    return influence


def environment_influence(env, snapshot, restore_fn, first_actions):
    """Return target-by-source real one-step influence magnitudes."""
    n_agents = first_actions.shape[1]
    base = first_actions[0].copy()
    base_next = one_step_from_snapshot(env, base, snapshot, restore_fn)
    influence = np.zeros((n_agents, n_agents), dtype=np.float64)
    perturbation_norms = np.zeros(n_agents, dtype=np.float64)
    for source in range(n_agents):
        alt_idx, distance = choose_alternative(first_actions, source)
        perturbed = base.copy()
        if distance <= 1e-8:
            perturbed[source] = np.clip(perturbed[source] + 0.05, -1.0, 1.0)
        else:
            perturbed[source] = first_actions[alt_idx, source]
        perturbation_norms[source] = np.linalg.norm(perturbed[source] - base[source])
        perturbed_next = one_step_from_snapshot(env, perturbed, snapshot, restore_fn)
        influence[:, source] = np.linalg.norm(perturbed_next - base_next, axis=-1)
    return influence, perturbation_norms


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def summarize_ranking(rows, scorer):
    selected = [row for row in rows if row["scorer"] == scorer]
    keys = (
        "top1",
        "top2",
        "spearman",
        "pairwise_acc",
        "regret",
        "normalized_regret",
        "selected_return",
        "oracle_return",
    )
    result = {"n_states": len(selected)}
    for key in keys:
        vals = np.asarray([row[key] for row in selected], dtype=np.float64)
        result[key] = float(np.nanmean(vals))
    result["selected_vs_random_gain"] = float(
        np.nanmean([row["selected_return"] - row["random_mean_return"] for row in selected])
    )
    return result


def summarize_influence(records, scorer):
    true_values = []
    pred_values = []
    per_state_cosine = []
    for rec in records:
        true = np.asarray(rec["true"], dtype=np.float64)
        pred = np.asarray(rec[scorer], dtype=np.float64)
        mask = ~np.eye(true.shape[0], dtype=bool)
        true_off = true[mask]
        pred_off = pred[mask]
        true_values.extend(true_off.tolist())
        pred_values.extend(pred_off.tolist())
        per_state_cosine.append(cosine_similarity(true_off, pred_off))
    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    scale = max(float(np.mean(np.abs(true_values))), 1e-12)
    finite_cosine = np.asarray(per_state_cosine, dtype=np.float64)
    finite_cosine = finite_cosine[np.isfinite(finite_cosine)]
    spearman = (
        corrcoef(rankdata(true_values), rankdata(pred_values))
        if np.std(pred_values) > 1e-12 else float("nan")
    )
    return {
        "n_offdiagonal_effects": int(len(true_values)),
        "true_effect_mean": float(np.mean(true_values)),
        "pred_effect_mean": float(np.mean(pred_values)),
        "pearson": corrcoef(true_values, pred_values),
        "spearman": spearman,
        "normalized_mae": float(np.mean(np.abs(pred_values - true_values)) / scale),
        "mean_state_cosine": (
            float(np.mean(finite_cosine)) if len(finite_cosine) else float("nan")
        ),
    }


def evaluate(args):
    if args.domain not in {"MPE", "MAMuJoCo"}:
        raise ValueError("Exact state restoration is available only for MPE and MAMuJoCo")

    run_dir = args.run_dir or run_dir_for(args.domain, args.task, args.quality)
    hcr_path = args.hcr_wm or wm_path(args.task, args.quality)
    mono_path = args.mono_wm or os.path.join(
        PROJECT_ROOT, "logs", "baseline_full_wm", "monolithic",
        args.task, args.quality, "best_model.pt",
    )
    ind_path = args.ind_wm or os.path.join(
        PROJECT_ROOT, "logs", "baseline_full_wm", "independent",
        args.task, args.quality, "best_model.pt",
    )
    required = {
        "policy run": run_dir,
        "HCR-WM": hcr_path,
        "monolithic WM": mono_path,
        "independent WM": ind_path,
    }
    for name, path in required.items():
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    data_n_agents, data_obs_dim, act_dim, discrete = infer_dims(
        args.domain, args.task, args.quality
    )
    if discrete:
        raise ValueError("This controlled influence evaluation currently excludes discrete SMAC")

    set_seed(args.seed)
    worker = load_worker(run_dir)
    device = worker.Config.device
    n_agents = int(worker.Config.n_agents)
    obs_dim = int(worker.normalizer.observation_dim)
    if n_agents != data_n_agents or obs_dim != data_obs_dim:
        print(
            "[DIMENSION OVERRIDE] "
            f"dataset=({data_n_agents} agents, obs {data_obs_dim}), "
            f"policy=({n_agents} agents, obs {obs_dim}); using policy dimensions",
            flush=True,
        )
    load_step = args.load_step if args.load_step > 0 else find_best_step(run_dir)
    load_policy_ckpt(worker, load_step)
    worker.Config.num_eval = 1
    worker.Config.num_envs = 1
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    models = {
        "hcr": load_wm(
            hcr_path, n_agents, obs_dim, act_dim, device,
            args.n_dyn, args.n_rew, args.hcr_hidden, args.n_slots,
        ),
        "monolithic": load_baseline(
            "monolithic", mono_path, n_agents, obs_dim, act_dim, device, args.baseline_hidden,
        ),
        "independent": load_baseline(
            "independent", ind_path, n_agents, obs_dim, act_dim, device, args.baseline_hidden,
        ),
    }

    config = worker.Config
    norm = worker.normalizer
    env = worker.env_list[0]
    test_ret = float(config.test_ret if args.test_ret is None else args.test_ret)
    returns = (test_ret * torch.ones(1, 1, config.n_agents)).to(device)
    env_ts = (torch.arange(config.horizon + config.history_horizon) - config.history_horizon).to(device)
    env_ts = env_ts.unsqueeze(0)
    obs = env.reset()[None]
    obs_queue = deque(maxlen=config.history_horizon + 1)
    if getattr(config, "use_zero_padding", False):
        obs_queue.extend([np.zeros_like(obs) for _ in range(config.history_horizon)])
    else:
        obs_queue.extend([norm.normalize(obs, "observations") for _ in range(config.history_horizon)])

    if args.domain == "MPE":
        snapshot_fn, restore_fn = snapshot_mpe_env, restore_mpe_env
    else:
        snapshot_fn, restore_fn = snapshot_mujoco_env, restore_mujoco_env

    ranking_rows = []
    influence_records = []
    episode_t = 0
    selfcheck_done = False

    for state_id in range(args.num_states):
        obs_queue.append(norm.normalize(obs, "observations"))
        raw_obs = obs.copy()
        first_acts, wm_acts, exec_acts = generate_candidates(
            worker, obs_queue, returns, env_ts, args.M, args.H, False, None
        )
        predicted = {
            name: to_np(wm_score(model, raw_obs, wm_acts, device)).reshape(args.M)
            for name, model in models.items()
        }

        snapshot = snapshot_fn(env)
        true_scores = np.asarray(
            [rollout_candidate(env, exec_acts[m, 0], False, snapshot, restore_fn) for m in range(args.M)],
            dtype=np.float64,
        )
        if not selfcheck_done:
            recheck = rollout_candidate(env, exec_acts[0, 0], False, snapshot, restore_fn)
            if abs(recheck - true_scores[0]) > 1e-5:
                raise RuntimeError(
                    f"State restore failed: first={true_scores[0]:.8f}, repeated={recheck:.8f}"
                )
            print(f"[RESTORE PASS] state={state_id} return={recheck:.8f}", flush=True)
            selfcheck_done = True

        true_best = int(np.argmax(true_scores))
        true_worst = int(np.argmin(true_scores))
        oracle_gap = float(true_scores[true_best] - true_scores[true_worst])
        for scorer, pred_scores in predicted.items():
            pred_best = int(np.argmax(pred_scores))
            pred_top2 = set(np.argsort(pred_scores)[-2:].tolist())
            regret = float(true_scores[true_best] - true_scores[pred_best])
            ranking_rows.append({
                "domain": args.domain,
                "task": args.task,
                "quality": args.quality,
                "state_id": state_id,
                "scorer": scorer,
                "M": args.M,
                "H": args.H,
                "test_ret": test_ret,
                "top1": int(pred_best == true_best),
                "top2": int(true_best in pred_top2),
                "spearman": corrcoef(rankdata(true_scores), rankdata(pred_scores)),
                "pairwise_acc": pairwise_acc(true_scores, pred_scores),
                "regret": regret,
                "normalized_regret": regret / max(oracle_gap, 1e-8),
                "selected_return": float(true_scores[pred_best]),
                "random_mean_return": float(np.mean(true_scores)),
                "oracle_return": float(true_scores[true_best]),
                "oracle_gap": oracle_gap,
            })

        first_exec = exec_acts[:, 0, 0]
        first_wm = wm_acts[:, 0, 0]
        true_influence, perturbation_norms = environment_influence(
            env, snapshot, restore_fn, first_exec
        )
        influence_records.append({
            "state_id": state_id,
            "true": true_influence.tolist(),
            "perturbation_norms": perturbation_norms.tolist(),
            **{
                name: model_influence(model, raw_obs, first_wm, device).tolist()
                for name, model in models.items()
            },
        })

        # Advance with candidate zero so anchor-state sampling is scorer-independent.
        restore_fn(env, snapshot)
        chosen = first_acts[0, 0]
        next_obs, reward, done, info = env.step(chosen)
        if getattr(config, "use_return_to_go", False):
            returns[0] = worker._update_return_to_go(returns[0], reward)
        episode_t += 1
        if bool(np.asarray(done).all()) or episode_t >= config.max_path_length - 1:
            obs = env.reset()[None]
            returns = (test_ret * torch.ones(1, 1, config.n_agents)).to(device)
            env_ts = (torch.arange(config.horizon + config.history_horizon) - config.history_horizon).to(device)
            env_ts = env_ts.unsqueeze(0)
            obs_queue.clear()
            if getattr(config, "use_zero_padding", False):
                obs_queue.extend([np.zeros_like(obs) for _ in range(config.history_horizon)])
            else:
                obs_queue.extend([norm.normalize(obs, "observations") for _ in range(config.history_horizon)])
            episode_t = 0
        else:
            obs = np.asarray(next_obs)[None]
            env_ts = env_ts + 1

        if (state_id + 1) % max(1, args.log_every) == 0:
            print(f"[PROGRESS] {state_id + 1}/{args.num_states}", flush=True)

    summary = {
        "protocol": {
            "domain": args.domain,
            "task": args.task,
            "quality": args.quality,
            "policy_run": run_dir,
            "load_step": int(load_step),
            "test_ret": test_ret,
            "steps": args.steps,
            "M": args.M,
            "H": args.H,
            "num_states": args.num_states,
            "anchor_advance": "candidate_0",
            "shared_candidates_across_scorers": True,
        },
        "checkpoints": {
            "hcr": hcr_path,
            "monolithic": mono_path,
            "independent": ind_path,
        },
        "ranking": {name: summarize_ranking(ranking_rows, name) for name in SCORERS},
        "cross_agent_influence": {
            name: summarize_influence(influence_records, name) for name in SCORERS
        },
    }
    return ranking_rows, influence_records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["MPE", "MAMuJoCo"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--hcr_wm", default="")
    parser.add_argument("--mono_wm", default="")
    parser.add_argument("--ind_wm", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_states", type=int, default=64)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--test_ret", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--n_dyn", type=int, default=8)
    parser.add_argument("--n_rew", type=int, default=4)
    parser.add_argument("--hcr_hidden", type=int, default=256)
    parser.add_argument("--baseline_hidden", type=int, default=512)
    parser.add_argument("--n_slots", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=8)
    parser.add_argument("--out_dir", default="results/hcr_new_evidence/same_state")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.makedirs(args.out_dir, exist_ok=True)

    rows, influence, summary = evaluate(args)
    stem = f"{args.domain}_{args.task}_{args.quality}".replace("/", "_")
    csv_path = os.path.join(args.out_dir, f"{stem}_ranking.csv")
    influence_path = os.path.join(args.out_dir, f"{stem}_influence.json")
    summary_path = os.path.join(args.out_dir, f"{stem}_summary.json")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(influence_path, "w") as handle:
        json.dump(influence, handle)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {influence_path}")
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
