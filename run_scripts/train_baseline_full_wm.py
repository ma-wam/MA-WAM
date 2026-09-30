import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.optim import Adam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from world_model.baseline_world_model import BaselineWorldModel
from run_scripts.paths import data_path_from_config


class TransitionDataset:
    def __init__(self, data_path, val_ratio=0.1, seed=42, discrete_action=False, num_actions=None):
        self.obs = np.load(os.path.join(data_path, "obs.npy"), mmap_mode="r")
        self.actions = np.load(os.path.join(data_path, "actions.npy"), mmap_mode="r")
        self.rewards = np.load(os.path.join(data_path, "rewards.npy"), mmap_mode="r")
        self.path_lengths = np.load(os.path.join(data_path, "path_lengths.npy"))
        self.discrete_action = discrete_action
        self.num_actions = num_actions

        valid = []
        start = 0
        for pl in self.path_lengths:
            pl = int(pl)
            if pl > 1:
                valid.append(np.arange(start, start + pl - 1, dtype=np.int64))
            start += pl
        valid = np.concatenate(valid)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(valid))
        n_val = int(len(valid) * val_ratio)
        self.val_idx = valid[perm[:n_val]]
        self.train_idx = valid[perm[n_val:]]
        self.rng = np.random.default_rng(seed + 1)
        print(f"[TransitionDataset] {len(self.train_idx)} train, {len(self.val_idx)} val transitions")

    def sample(self, split, batch_size, device):
        pool = self.train_idx if split == "train" else self.val_idx
        idx = self.rng.choice(pool, batch_size, replace=False)
        obs = np.asarray(self.obs[idx], dtype=np.float32)
        next_obs = np.asarray(self.obs[idx + 1], dtype=np.float32)
        act = np.asarray(self.actions[idx])
        rew = np.asarray(self.rewards[idx], dtype=np.float32)

        if self.discrete_action and self.num_actions is not None:
            if act.ndim == 3 and act.shape[-1] == 1:
                act = act[..., 0]
            n, n_agents = act.shape[:2]
            onehot = np.zeros((n, n_agents, self.num_actions), dtype=np.float32)
            np.put_along_axis(onehot, act.astype(np.int64)[..., None], 1.0, axis=-1)
            act = onehot
        else:
            act = act.astype(np.float32)

        if rew.ndim == 2:
            rew = rew[..., None]

        return {
            "obs": torch.as_tensor(obs, device=device),
            "act": torch.as_tensor(act, device=device),
            "next_obs": torch.as_tensor(next_obs, device=device),
            "reward": torch.as_tensor(rew, device=device),
        }


class Trainer:
    def __init__(self, model, dataset, batch_size, lr, device):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.opt = Adam(self.model.parameters(), lr=lr)

    def train(self, n_steps, save_dir, save_freq=5000, log_freq=500):
        os.makedirs(save_dir, exist_ok=True)
        best = float("inf")
        t0 = time.time()
        for step in range(1, n_steps + 1):
            self.model.train()
            batch = self.dataset.sample("train", self.batch_size, self.device)
            loss, info = self.model.loss(batch["obs"], batch["act"], batch["next_obs"], batch["reward"])
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.opt.step()

            if step % log_freq == 0:
                speed = step / max(time.time() - t0, 1e-6)
                print(
                    f"[Step {step}/{n_steps}] dyn_loss={info['dynamics_loss']:.4f} "
                    f"rew_loss={info['reward_loss']:.4f} total={info['total_loss']:.4f} "
                    f"({speed:.1f} steps/s)",
                    flush=True,
                )

            if step % save_freq == 0:
                val = self.validate()
                print(f"  [Validation] loss={val:.4f} (best={best:.4f})", flush=True)
                ckpt = {
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "val_loss": val,
                    "model_type": self.model.model_type,
                    "hidden_dim": self.model.hidden_dim,
                }
                torch.save(ckpt, os.path.join(save_dir, f"wm_step_{step}.pt"))
                if val < best:
                    best = val
                    torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
                    print(f"  [New best model saved] val_loss={val:.4f}", flush=True)
        print(f"Training complete. Best validation loss: {best:.4f}", flush=True)

    @torch.no_grad()
    def validate(self, n_batches=20):
        self.model.eval()
        vals = []
        for _ in range(n_batches):
            batch = self.dataset.sample("val", self.batch_size, self.device)
            loss, _ = self.model.loss(batch["obs"], batch["act"], batch["next_obs"], batch["reward"])
            vals.append(float(loss.item()))
        return float(np.mean(vals))


def resolve_data_path(config):
    return data_path_from_config(config, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-g", "--gpu", default="0")
    ap.add_argument("--model", choices=["monolithic", "independent"], required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--save_root", default="logs/baseline_full_wm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    discrete_action = config.get("discrete_action", False)
    num_actions = config.get("num_actions", None)
    act_dim = num_actions if discrete_action and num_actions else config["act_dim"]
    data_path = resolve_data_path(config)
    print("=" * 70)
    print(f"Training full baseline WM: {args.model}")
    print(f"Config: {args.config}")
    print(f"Data: {data_path}")
    print(f"Device: {device}, hidden={args.hidden}")

    model = BaselineWorldModel(
        n_agents=config["n_agents"],
        obs_dim=config["obs_dim"],
        act_dim=act_dim,
        hidden_dim=args.hidden,
        model_type=args.model,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    ds = TransitionDataset(
        data_path,
        val_ratio=config.get("val_ratio", 0.1),
        seed=args.seed,
        discrete_action=discrete_action,
        num_actions=num_actions,
    )
    trainer = Trainer(
        model=model,
        dataset=ds,
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 3e-4),
        device=device,
    )
    save_dir = os.path.join(
        PROJECT_ROOT,
        args.save_root,
        args.model,
        config["env_name"],
        config["data_split"],
    )
    trainer.train(
        n_steps=args.steps or config.get("n_train_steps", 40000),
        save_dir=save_dir,
        save_freq=config.get("save_freq", 5000),
        log_freq=config.get("log_freq", 500),
    )
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
