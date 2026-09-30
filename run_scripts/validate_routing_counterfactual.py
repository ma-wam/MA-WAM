from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from run_scripts.paths import data_path_from_config  # noqa: E402


def load_model(project: Path, config: dict, ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(project))
    from world_model import MoEWorldModel  # noqa: WPS433

    model = MoEWorldModel(
        n_agents=config["n_agents"],
        obs_dim=config["obs_dim"],
        act_dim=config.get("num_actions") if config.get("discrete_action", False) else config["act_dim"],
        n_dyn_experts=config.get("n_dyn_experts", 8),
        n_rew_experts=config.get("n_rew_experts", 4),
        hidden_dim=config.get("hidden_dim", 256),
        n_slots=config.get("n_slots", 4),
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def resolve_data_path(project: Path, config: dict) -> Path:
    return Path(data_path_from_config(config, project))


def sample_indices(path_lengths: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    starts = []
    offset = 0
    for length in path_lengths:
        length = int(length)
        if length > 1:
            starts.append(np.arange(offset, offset + length - 1, dtype=np.int64))
        offset += length
    starts = np.concatenate(starts)
    rng = np.random.default_rng(seed)
    if len(starts) > n_samples:
        starts = rng.choice(starts, size=n_samples, replace=False)
    return starts


def onehot_actions(actions: np.ndarray, num_actions: int) -> np.ndarray:
    n, n_agents = actions.shape[:2]
    out = np.zeros((n, n_agents, num_actions), dtype=np.float32)
    np.put_along_axis(out, actions.astype(np.int64)[..., None], 1.0, axis=-1)
    return out


def upper_values(matrix: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            vals.append(matrix[i, j])
    return np.asarray(vals, dtype=np.float64)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def analyze_one(project: Path, cfg_path: Path, ckpt_path: Path, out_dir: Path, n_samples: int, seed: int, device: torch.device) -> dict:
    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)
    model = load_model(project, config, ckpt_path, device)
    data_path = resolve_data_path(project, config)
    obs = np.load(data_path / "obs.npy", mmap_mode="r")
    actions = np.load(data_path / "actions.npy", mmap_mode="r")
    path_lengths = np.load(data_path / "path_lengths.npy")
    idx = sample_indices(path_lengths, n_samples, seed)

    obs_np = np.asarray(obs[idx], dtype=np.float32)
    act_np = np.asarray(actions[idx])
    if config.get("discrete_action", False):
        act_np = onehot_actions(act_np, config["num_actions"])
    else:
        act_np = act_np.astype(np.float32)

    obs_t = torch.as_tensor(obs_np, device=device)
    act_t = torch.as_tensor(act_np, device=device)
    pred, info = model.dynamics.forward_with_routing(obs_t, act_t)

    n_agents = config["n_agents"]
    n_experts = config.get("n_dyn_experts", 8)
    n_slots = config.get("n_slots", 4)
    combine = info["combine_probs"].reshape(len(idx), n_agents, n_experts, n_slots).sum(dim=-1)
    routing_vec = combine.mean(dim=0).detach().cpu().numpy()
    routing = np.zeros((n_agents, n_agents), dtype=np.float64)
    for i in range(n_agents):
        for j in range(n_agents):
            a, b = routing_vec[i], routing_vec[j]
            routing[i, j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    rng = np.random.default_rng(seed + 13)
    action_scale = np.std(act_np.reshape(-1, act_np.shape[-1]), axis=0).astype(np.float32)
    action_scale = np.maximum(action_scale, 1e-3)
    influence = np.zeros((n_agents, n_agents), dtype=np.float64)
    for source in range(n_agents):
        pert = act_np.copy()
        if config.get("discrete_action", False):
            choices = rng.integers(0, config["num_actions"], size=len(idx))
            pert[:, source, :] = 0.0
            pert[np.arange(len(idx)), source, choices] = 1.0
        else:
            noise = rng.normal(0, 0.35, size=(len(idx), act_np.shape[-1])).astype(np.float32) * action_scale
            pert[:, source, :] += noise
        pert_t = torch.as_tensor(pert, device=device)
        pred_pert = model.predict_next_obs(obs_t, pert_t)
        delta = (pred_pert - pred).detach().cpu().numpy()
        for target in range(n_agents):
            influence[target, source] = float(np.linalg.norm(delta[:, target, :], axis=-1).mean())
    influence_sym = 0.5 * (influence + influence.T)
    np.fill_diagonal(influence_sym, 0.0)

    x = upper_values(routing)
    y = upper_values(influence_sym)
    result = {
        "env_name": config["env_name"],
        "split": config["data_split"],
        "n_agents": n_agents,
        "n_samples": int(len(idx)),
        "routing_counterfactual_corr": corr(x, y),
        "routing_upper": x.tolist(),
        "influence_upper": y.tolist(),
        "routing_matrix": routing.tolist(),
        "counterfactual_influence_matrix": influence_sym.tolist(),
        "config": str(cfg_path),
        "ckpt": str(ckpt_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{config['env_name']}_{config['data_split']}".replace("/", "_")
    with (out_dir / f"{name}_routing_counterfactual.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("results/routing_counterfactual"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(args.device)
    jobs = [
        ("exp_specs/phase1_wm/mamujoco/2ant/wm_2ant_good.yaml", "logs/phase1_wm/2ant/Good/best_model.pt"),
        ("exp_specs/phase1_wm/mamujoco/4ant/wm_4ant_good.yaml", "logs/phase1_wm/4ant/Good/best_model.pt"),
    ]
    rows = []
    for cfg, ckpt in jobs:
        rows.append(analyze_one(args.project, args.project / cfg, args.project / ckpt, args.project / args.out_dir, args.n_samples, 7, device))
    fields = ["env_name", "split", "n_agents", "n_samples", "routing_counterfactual_corr"]
    with (args.project / args.out_dir / "routing_counterfactual_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
