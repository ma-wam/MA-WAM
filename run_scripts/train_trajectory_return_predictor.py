"""Train a direct H-step trajectory-return predictor on offline episodes.

The train/validation split is made at the episode level before windows are
sampled.  Consequently no H-step window crosses an episode boundary and no
episode contributes to both splits.

Example:
  python run_scripts/train_trajectory_return_predictor.py \
    -c exp_specs/phase1_wm/mpe/simple_spread/wm_spread_medium.yaml \
    -g 0 --horizon 8
"""

from __future__ import annotations

from training_seeds import add_training_arguments, prepare_training, seed_output
import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml
from torch.optim import Adam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from world_model.trajectory_return_predictor import TrajectoryReturnPredictor
from run_scripts.paths import data_path_from_config


def resolve_data_path(config: dict) -> str:
    return data_path_from_config(config, PROJECT_ROOT)


@dataclass(frozen=True)
class WindowSplit:
    episode_ids: np.ndarray
    valid_counts: np.ndarray
    cumulative_counts: np.ndarray
    total_windows: int


class EpisodeWindowSampler:
    """Memory-mapped sampler of fixed-length windows from complete episodes."""

    def __init__(
        self,
        data_path: str,
        horizon: int,
        val_ratio: float,
        split_seed: int,
        discrete_action: bool,
        num_actions: Optional[int],
        expected_n_agents: int,
        expected_obs_dim: int,
        expected_act_dim: int,
    ) -> None:
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
        self.data_path = os.path.abspath(data_path)
        self.horizon = int(horizon)
        self.discrete_action = bool(discrete_action)
        self.num_actions = int(num_actions) if num_actions is not None else None

        def load(name: str):
            path = os.path.join(self.data_path, name)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return np.load(path, mmap_mode="r")

        self.observations = load("obs.npy")
        self.actions = load("actions.npy")
        self.rewards = load("rewards.npy")
        self.path_lengths = np.asarray(load("path_lengths.npy"), dtype=np.int64)
        self._validate_shapes(
            expected_n_agents, expected_obs_dim, expected_act_dim
        )

        self.episode_starts = np.zeros(len(self.path_lengths), dtype=np.int64)
        if len(self.path_lengths) > 1:
            self.episode_starts[1:] = np.cumsum(self.path_lengths[:-1])

        rng = np.random.default_rng(split_seed)
        episode_order = rng.permutation(len(self.path_lengths))
        n_val = max(1, int(round(len(episode_order) * val_ratio)))
        n_val = min(n_val, len(episode_order) - 1)
        self.splits = {
            "val": self._make_split(episode_order[:n_val]),
            "train": self._make_split(episode_order[n_val:]),
        }

    @property
    def n_agents(self) -> int:
        return int(self.observations.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[2])

    @property
    def act_dim(self) -> int:
        if self.discrete_action:
            assert self.num_actions is not None
            return self.num_actions
        return int(self.actions.shape[-1])

    def _validate_shapes(
        self, expected_n_agents: int, expected_obs_dim: int, expected_act_dim: int
    ) -> None:
        if self.observations.ndim != 3:
            raise ValueError(f"obs.npy must be (T,A,obs_dim), got {self.observations.shape}")
        total = int(self.path_lengths.sum())
        lengths = [len(self.observations), len(self.actions), len(self.rewards)]
        if len(set(lengths)) != 1 or total != lengths[0]:
            raise ValueError(
                "Episode/data length mismatch: "
                f"sum(path_lengths)={total}, obs/actions/rewards={lengths}"
            )
        actual_agents, actual_obs_dim = self.observations.shape[1:]
        if (actual_agents, actual_obs_dim) != (
            expected_n_agents,
            expected_obs_dim,
        ):
            raise ValueError(
                "Config/data observation mismatch: "
                f"config A={expected_n_agents}, obs_dim={expected_obs_dim}; "
                f"data shape={self.observations.shape}. Use the matching MPE n-agent config."
            )
        if self.rewards.ndim not in (2, 3) or self.rewards.shape[1] != actual_agents:
            raise ValueError(
                f"rewards.npy must start with (T,{actual_agents}), got {self.rewards.shape}"
            )
        if self.discrete_action:
            if self.num_actions is None or self.num_actions <= 1:
                raise ValueError("Discrete data requires num_actions > 1")
            action_shape = self.actions.shape
            if self.actions.ndim == 3 and self.actions.shape[-1] == 1:
                action_shape = self.actions.shape[:-1]
            if len(action_shape) != 2 or action_shape[1] != actual_agents:
                raise ValueError(
                    f"Discrete actions must be (T,A) integer indices, got {self.actions.shape}"
                )
            # Cheap range validation on the memory map without copying it all.
            stride = max(1, len(self.actions) // 100_000)
            sample = np.asarray(self.actions[::stride]).reshape(-1)
            if sample.min() < 0 or sample.max() >= self.num_actions:
                raise ValueError(
                    f"Discrete action index outside [0,{self.num_actions - 1}]"
                )
        else:
            expected = (len(self.observations), actual_agents, expected_act_dim)
            if tuple(self.actions.shape) != expected:
                raise ValueError(
                    f"Continuous action mismatch: expected {expected}, got {self.actions.shape}"
                )

    def _make_split(self, episode_ids: np.ndarray) -> WindowSplit:
        episode_ids = np.asarray(episode_ids, dtype=np.int64)
        valid_counts = np.maximum(
            self.path_lengths[episode_ids] - self.horizon + 1, 0
        ).astype(np.int64)
        keep = valid_counts > 0
        episode_ids = episode_ids[keep]
        valid_counts = valid_counts[keep]
        cumulative = np.cumsum(valid_counts)
        total = int(cumulative[-1]) if len(cumulative) else 0
        if total == 0:
            raise ValueError(
                f"No episodes in split are long enough for horizon={self.horizon}"
            )
        return WindowSplit(episode_ids, valid_counts, cumulative, total)

    def sample_numpy(
        self, split: str, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info = self.splits[split]
        flat_ids = rng.integers(0, info.total_windows, size=batch_size)
        positions = np.searchsorted(info.cumulative_counts, flat_ids, side="right")
        previous = np.zeros(batch_size, dtype=np.int64)
        nonzero = positions > 0
        previous[nonzero] = info.cumulative_counts[positions[nonzero] - 1]
        offsets = flat_ids - previous
        episode_ids = info.episode_ids[positions]
        starts = self.episode_starts[episode_ids] + offsets
        indices = starts[:, None] + np.arange(self.horizon, dtype=np.int64)[None]

        obs = np.asarray(self.observations[starts], dtype=np.float32)
        actions = np.asarray(self.actions[indices])
        if self.discrete_action:
            if actions.ndim == 4 and actions.shape[-1] == 1:
                actions = actions[..., 0]
            action_ids = actions.astype(np.int64, copy=False)
            actions = np.eye(self.num_actions, dtype=np.float32)[action_ids]
        else:
            actions = actions.astype(np.float32, copy=False)

        rewards = np.asarray(self.rewards[indices], dtype=np.float32)
        returns = rewards.reshape(batch_size, -1).sum(axis=1, dtype=np.float32)
        return obs, actions, returns

    def sample_torch(
        self,
        split: str,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, actions, returns = self.sample_numpy(split, batch_size, rng)
        return (
            torch.as_tensor(obs, device=device),
            torch.as_tensor(actions, device=device),
            torch.as_tensor(returns, device=device),
        )

    def estimate_target_stats(self, n_samples: int, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        remaining = int(n_samples)
        count, total, total_sq = 0, 0.0, 0.0
        while remaining > 0:
            batch = min(4096, remaining)
            _, _, returns = self.sample_numpy("train", batch, rng)
            values = returns.astype(np.float64)
            count += len(values)
            total += float(values.sum())
            total_sq += float(np.square(values).sum())
            remaining -= batch
        mean = total / count
        variance = max(total_sq / count - mean * mean, 1e-12)
        return mean, float(np.sqrt(variance))

    def metadata(self) -> dict:
        return {
            "data_path": self.data_path,
            "horizon": self.horizon,
            "n_episodes": int(len(self.path_lengths)),
            "train_episodes": int(len(self.splits["train"].episode_ids)),
            "val_episodes": int(len(self.splits["val"].episode_ids)),
            "train_windows": self.splits["train"].total_windows,
            "val_windows": self.splits["val"].total_windows,
            "discrete_action": self.discrete_action,
            "num_actions": self.num_actions,
        }


@torch.no_grad()
def validate(
    model: TrajectoryReturnPredictor,
    sampler: EpisodeWindowSampler,
    batch_size: int,
    n_batches: int,
    seed: int,
    device: torch.device,
) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    predictions, targets = [], []
    normalized_mse = 0.0
    for _ in range(n_batches):
        obs, actions, target = sampler.sample_torch(
            "val", batch_size, rng, device
        )
        normalized_mse += float(model.loss(obs, actions, target).item())
        predictions.append(model(obs, actions).cpu().numpy())
        targets.append(target.cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    correlation = 0.0
    if prediction.std() > 1e-12 and target.std() > 1e-12:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "normalized_mse": normalized_mse / n_batches,
        "raw_mse": float(np.mean((prediction - target) ** 2)),
        "raw_mae": float(np.mean(np.abs(prediction - target))),
        "pearson": correlation,
    }


def save_checkpoint(
    path: str,
    model: TrajectoryReturnPredictor,
    optimizer: Adam,
    step: int,
    val_metrics: dict,
    config: dict,
    sampler: EpisodeWindowSampler,
    split_seed: int,
) -> None:
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "model_config": model.model_config(),
        "data_config": sampler.metadata(),
        "episode_split_seed": split_seed,
        "source_config": config,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_trp_", suffix=".pt", dir=directory
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=20260715)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--stats_samples", type=int, default=100_000)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--log_freq", type=int, default=None)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_dir", default="")
    add_training_arguments(parser)
    args = parser.parse_args()
    if prepare_training(args, 'train_seed'):
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    discrete = bool(config.get("discrete_action", False))
    num_actions = config.get("num_actions") if discrete else None
    act_dim = int(num_actions if discrete else config["act_dim"])
    data_path = resolve_data_path(config)
    val_ratio = float(
        args.val_ratio if args.val_ratio is not None else config.get("val_ratio", 0.1)
    )
    sampler = EpisodeWindowSampler(
        data_path=data_path,
        horizon=args.horizon,
        val_ratio=val_ratio,
        split_seed=args.split_seed,
        discrete_action=discrete,
        num_actions=num_actions,
        expected_n_agents=int(config["n_agents"]),
        expected_obs_dim=int(config["obs_dim"]),
        expected_act_dim=act_dim,
    )

    model = TrajectoryReturnPredictor(
        n_agents=sampler.n_agents,
        obs_dim=sampler.obs_dim,
        act_dim=sampler.act_dim,
        horizon=args.horizon,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    target_mean, target_std = sampler.estimate_target_stats(
        max(1, args.stats_samples), args.split_seed + 1
    )
    model.set_target_stats(target_mean, target_std)

    steps = int(
        args.steps
        if args.steps is not None
        else config.get("trp_train_steps", config.get("n_train_steps", 40_000))
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 256)
    )
    lr = float(args.lr if args.lr is not None else config.get("learning_rate", 3e-4))
    save_freq = int(
        args.save_freq
        if args.save_freq is not None
        else config.get("save_freq", 5_000)
    )
    log_freq = int(
        args.log_freq if args.log_freq is not None else config.get("log_freq", 500)
    )
    save_dir = args.save_dir or os.path.join(
        PROJECT_ROOT,
        "logs",
        "trajectory_return_predictor",
        config["env_name"],
        str(config["data_split"]),
        f"H{args.horizon}",
    )
    save_dir = seed_output(save_dir, args)
    os.makedirs(save_dir, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(args.train_seed)

    print(f"Direct trajectory-return predictor | device={device}")
    print(f"Config: {args.config}")
    print(f"Data: {json.dumps(sampler.metadata(), ensure_ascii=False)}")
    print(
        f"Model: {model.model_config()} params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Target(train only): mean={target_mean:.6f} std={target_std:.6f}")
    print(
        f"Optimization: steps={steps} batch={batch_size} lr={lr} "
        f"split_seed={args.split_seed} train_seed={args.train_seed}"
    )

    best = float("inf")
    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        obs, actions, target = sampler.sample_torch(
            "train", batch_size, rng, device
        )
        loss = model.loss(obs, actions, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step == 1 or step % log_freq == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[Step {step}/{steps}] normalized_mse={loss.item():.6f} "
                f"steps_per_sec={step / elapsed:.2f}"
            )

        if step % save_freq == 0 or step == steps:
            metrics = validate(
                model,
                sampler,
                batch_size,
                args.val_batches,
                args.split_seed + 2,
                device,
            )
            print(f"[Validation step={step}] {json.dumps(metrics)}")
            checkpoint_path = os.path.join(save_dir, f"trp_step_{step}.pt")
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                metrics,
                config,
                sampler,
                args.split_seed,
            )
            if metrics["normalized_mse"] < best:
                best = metrics["normalized_mse"]
                save_checkpoint(
                    os.path.join(save_dir, "best_model.pt"),
                    model,
                    optimizer,
                    step,
                    metrics,
                    config,
                    sampler,
                    args.split_seed,
                )
                print(f"[New best] normalized_mse={best:.6f}")

    print(f"Done. best_normalized_mse={best:.6f} save_dir={save_dir}")


if __name__ == "__main__":
    main()
