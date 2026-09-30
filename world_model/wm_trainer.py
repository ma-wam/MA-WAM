"""
世界模型训练器 - 在离线数据上训练 MoE 世界模型

从 M2Flow 格式的 npy 数据中提取转移元组 (s_t, a_t, s_{t+1}, r_t)，
进行标准监督学习训练。
"""

import os
import time

import numpy as np
import torch
from torch.optim import Adam

from .world_model import MoEWorldModel


class WorldModelTrainer:
    """
    世界模型训练器

    Args:
        world_model: MoEWorldModel 实例
        data_path: 数据目录 (包含 obs.npy, actions.npy, rewards.npy, path_lengths.npy)
        batch_size: 批大小
        lr: 学习率
        device: 计算设备
        val_ratio: 验证集比例
    """

    def __init__(
        self,
        world_model,
        data_path,
        batch_size=256,
        lr=3e-4,
        device="cuda",
        val_ratio=0.1,
        use_wandb=False,
        wandb_project="CoFlow-WM",
        wandb_run_name=None,
        wandb_config=None,
        discrete_action=False,
        num_actions=None,
    ):
        self.world_model = world_model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimizer = Adam(world_model.parameters(), lr=lr)

        # wandb
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config or {})
                self._wandb = wandb
                print(f"[wandb] Initialized: project={wandb_project}, run={wandb_run_name}")
            except Exception as e:
                print(f"[wandb] Init failed: {e}")
                self.use_wandb = False

        # 离散动作处理
        self.discrete_action = discrete_action
        self.num_actions = num_actions

        # 加载并准备转移数据
        self._prepare_data(data_path, val_ratio)

    def _prepare_data(self, data_path, val_ratio):
        """从 episode 数据中提取转移元组"""
        obs = np.load(os.path.join(data_path, "obs.npy"))
        actions = np.load(os.path.join(data_path, "actions.npy"))
        rewards = np.load(os.path.join(data_path, "rewards.npy"))
        path_lengths = np.load(os.path.join(data_path, "path_lengths.npy"))

        # 提取有效转移 (s_t, a_t, s_{t+1}, r_t)
        obs_t_list, act_t_list, obs_tp1_list, rew_t_list = [], [], [], []

        start = 0
        for pl in path_lengths:
            end = start + pl
            # 每个 episode 中的转移: t -> t+1
            for t in range(int(pl) - 1):
                obs_t_list.append(obs[start + t])
                act_t_list.append(actions[start + t])
                obs_tp1_list.append(obs[start + t + 1])
                rew_t_list.append(rewards[start + t])
            start = end

        self.obs_t = np.array(obs_t_list, dtype=np.float32)
        self.obs_tp1 = np.array(obs_tp1_list, dtype=np.float32)
        self.rew_t = np.array(rew_t_list, dtype=np.float32)

        # 离散动作 one-hot 编码
        act_arr = np.array(act_t_list)
        if self.discrete_action and self.num_actions is not None:
            # act_arr: (N, n_agents) int -> (N, n_agents, num_actions) float32
            N, n_ag = act_arr.shape
            one_hot = np.zeros((N, n_ag, self.num_actions), dtype=np.float32)
            for i in range(n_ag):
                one_hot[np.arange(N), i, act_arr[:, i].astype(int)] = 1.0
            self.act_t = one_hot
            print(f"[WM Trainer] Discrete actions: one-hot encoded to shape {self.act_t.shape}")
        else:
            self.act_t = act_arr.astype(np.float32)

        # 处理 reward 形状
        if self.rew_t.ndim == 2:
            self.rew_t = self.rew_t[..., np.newaxis]  # (N, n_agents, 1)

        n_total = len(self.obs_t)
        n_val = int(n_total * val_ratio)

        # 随机 shuffle 并划分
        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.train_indices = train_indices
        self.val_indices = val_indices

        print(f"[WorldModelTrainer] Prepared {len(train_indices)} train, {len(val_indices)} val transitions")

    def _sample_batch(self, split="train"):
        """采样一个 batch"""
        indices = self.train_indices if split == "train" else self.val_indices
        batch_idx = np.random.choice(indices, self.batch_size, replace=False)

        batch = {
            "obs": torch.tensor(self.obs_t[batch_idx], device=self.device),
            "act": torch.tensor(self.act_t[batch_idx], device=self.device),
            "next_obs": torch.tensor(self.obs_tp1[batch_idx], device=self.device),
            "reward": torch.tensor(self.rew_t[batch_idx], device=self.device),
        }
        return batch

    def train(self, n_steps=40000, save_dir="logs/phase1_wm", save_freq=5000, log_freq=500):
        """
        训练世界模型

        Args:
            n_steps: 总训练步数
            save_dir: checkpoint 保存目录
            save_freq: checkpoint 保存频率
            log_freq: 日志频率
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")
        start_time = time.time()

        for step in range(1, n_steps + 1):
            self.world_model.train()
            batch = self._sample_batch("train")

            loss, info = self.world_model.loss(
                batch["obs"], batch["act"], batch["next_obs"], batch["reward"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=10.0)
            self.optimizer.step()

            # 日志
            if step % log_freq == 0:
                elapsed = time.time() - start_time
                fps = step / elapsed
                print(
                    f"[Step {step}/{n_steps}] "
                    f"dyn_loss={info['dynamics_loss']:.4f} "
                    f"rew_loss={info['reward_loss']:.4f} "
                    f"total={info['total_loss']:.4f} "
                    f"({fps:.1f} steps/s)"
                )
                if self.use_wandb and self._wandb:
                    self._wandb.log({"train/" + k: v for k, v in info.items()}, step=step)

            # 验证
            if step % save_freq == 0:
                val_loss = self._validate()
                print(f"  [Validation] loss={val_loss:.4f} (best={best_val_loss:.4f})")

                # 保存 checkpoint
                ckpt_path = os.path.join(save_dir, f"wm_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": self.world_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }, ckpt_path)

                if self.use_wandb and self._wandb:
                    self._wandb.log({"val/loss": val_loss, "val/best_loss": best_val_loss}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(save_dir, "best_model.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": self.world_model.state_dict(),
                        "val_loss": val_loss,
                    }, best_path)
                    print(f"  [New best model saved] val_loss={val_loss:.4f}")

        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

    @torch.no_grad()
    def _validate(self, n_batches=20):
        """计算验证损失"""
        self.world_model.eval()
        total_loss = 0.0

        for _ in range(n_batches):
            batch = self._sample_batch("val")
            _, info = self.world_model.loss(
                batch["obs"], batch["act"], batch["next_obs"], batch["reward"]
            )
            total_loss += info["total_loss"]

        return total_loss / n_batches
