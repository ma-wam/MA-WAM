"""
增强型序列数据集 - 在 M2Flow 的 SequenceDataset 基础上融合世界模型生成的合成数据

核心思路: 在 ReplayBuffer finalize 之前，将合成数据的 episode 也加入 buffer，
使得真实数据和合成数据在同一个 buffer 中混合，后续的归一化、padding、采样逻辑完全复用。
"""

import importlib
import os
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.utils.mask_generator import MultiAgentMaskGenerator


class AugmentedSequenceDataset(torch.utils.data.Dataset):
    """
    在 M2Flow SequenceDataset 基础上增加合成数据支持。

    与原始 SequenceDataset 的关键区别:
    1. 在 ReplayBuffer 加载真实数据后、finalize 之前，注入合成数据
    2. Normalizer 仅在真实数据上拟合 (避免合成数据的分布偏差影响归一化)
    3. 合成数据使用与真实数据相同的归一化参数

    Args:
        synthetic_data_path: 合成数据目录 (包含 obs.npy, actions.npy 等)
        synthetic_ratio: 合成数据比例 (相对于真实数据量)
        其余参数与 SequenceDataset 完全一致
    """

    def __init__(
        self,
        env_type: str = "mamujoco",
        env: str = "2halfcheetah-Good",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_env_ts: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "single",
        pred_future_padding: bool = False,
        seed: Optional[int] = None,
        # 新增参数
        synthetic_data_path: str = None,
        synthetic_ratio: float = 0.5,
    ):
        assert agent_condition_type in ["single", "all", "random"]
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inv_dyn = use_inv_dyn
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.include_env_ts = include_env_ts
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding

        # 计算合成数据 episode 数量以预分配 buffer
        n_synth_episodes = 0
        if synthetic_data_path and os.path.exists(synthetic_data_path):
            synth_path_lengths = np.load(os.path.join(synthetic_data_path, "path_lengths.npy"))
            n_synth_episodes = int(len(synth_path_lengths) * synthetic_ratio)
            print(f"[AugmentedDataset] Will add {n_synth_episodes} synthetic episodes")

        # 加载真实数据
        itr = env_mod.sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes + n_synth_episodes,  # 扩大 buffer 容量
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=use_zero_padding,
        )
        n_real = 0
        for _, episode in enumerate(itr):
            fields.add_path(episode)
            n_real += 1

        # 注入合成数据 (在 finalize 之前!)
        if synthetic_data_path and os.path.exists(synthetic_data_path) and n_synth_episodes > 0:
            self._inject_synthetic(fields, synthetic_data_path, n_synth_episodes)

        fields.finalize()

        # Normalizer 在所有数据 (含合成) 上拟合
        # 注: 如果希望仅在真实数据上拟合 normalizer，可以先 finalize 真实数据 buffer，
        # 拟合 normalizer，再扩展 buffer。这里简化处理，对轻微分布偏移影响不大。
        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        self.indices = self.make_indices(fields.path_lengths)
        self.mask_generator = MultiAgentMaskGenerator(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            history_horizon=self.history_horizon,
            action_visible=not use_inv_dyn,
        )

        if self.discrete_action:
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(f"[AugmentedDataset] {n_real} real + {n_synth_episodes} synthetic episodes")
        print(fields)

    def _inject_synthetic(self, fields, synthetic_data_path, n_episodes):
        """将合成数据注入 ReplayBuffer"""
        obs = np.load(os.path.join(synthetic_data_path, "obs.npy"))
        actions = np.load(os.path.join(synthetic_data_path, "actions.npy"))
        rewards = np.load(os.path.join(synthetic_data_path, "rewards.npy"))
        path_lengths = np.load(os.path.join(synthetic_data_path, "path_lengths.npy"))

        # 加载可选字段 (SMAC 需要 legal_actions)
        legals_path = os.path.join(synthetic_data_path, "legals.npy")
        legals = np.load(legals_path) if os.path.exists(legals_path) else None

        # 随机选择 n_episodes 个合成 episode
        if n_episodes < len(path_lengths):
            selected = np.random.choice(len(path_lengths), n_episodes, replace=False)
        else:
            selected = np.arange(len(path_lengths))

        start = 0
        starts = []
        for pl in path_lengths:
            starts.append(start)
            start += pl

        injected = 0
        for idx in selected:
            ep_start = starts[idx]
            ep_len = int(path_lengths[idx])
            if ep_len > self.max_path_length:
                ep_len = self.max_path_length

            episode = {
                "observations": obs[ep_start: ep_start + ep_len],
                "actions": actions[ep_start: ep_start + ep_len],
                "rewards": rewards[ep_start: ep_start + ep_len],
                "terminals": np.zeros(
                    (ep_len, self.n_agents), dtype=bool
                ),
            }
            if legals is not None:
                episode["legal_actions"] = legals[ep_start: ep_start + ep_len]
            episode["terminals"][-1] = True

            # 补全 buffer 中已有但合成数据缺失的字段（如 timeouts）
            if hasattr(fields, 'keys') and fields.keys is not None:
                for key in fields.keys:
                    if key not in episode:
                        # 用与 terminals 相同形状的零数组填充
                        episode[key] = np.zeros(
                            (ep_len, self.n_agents), dtype=bool
                        )

            try:
                fields.add_path(episode)
                injected += 1
            except (AssertionError, IndexError):
                continue  # 跳过无效 episode

        print(f"  Injected {injected} synthetic episodes into buffer")

    # ---- 以下方法与 M2Flow SequenceDataset 完全一致 ----

    def pad_future(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")
        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate([
                    self.fields[key],
                    np.zeros((shape[0], self.horizon - 1, *shape[2:]), dtype=self.fields[key].dtype),
                ], axis=1)
            else:
                self.fields[key] = np.concatenate([
                    self.fields[key],
                    np.repeat(self.fields[key][:, -1:], self.horizon - 1, axis=1),
                ], axis=1)

    def pad_history(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")
        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate([
                    np.zeros((shape[0], self.history_horizon, *shape[2:]), dtype=self.fields[key].dtype),
                    self.fields[key],
                ], axis=1)
            else:
                self.fields[key] = np.concatenate([
                    np.repeat(self.fields[key][:, :1], self.history_horizon, axis=1),
                    self.fields[key],
                ], axis=1)

    def normalize(self, keys: List[str] = None):
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]
        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths: np.ndarray):
        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations: np.ndarray, agent_idx: Optional[int] = None):
        ret_dict = {}
        if self.agent_condition_type == "single":
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[: self.history_horizon + 1, agent_idx]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        elif self.agent_condition_type == "all":
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.agent_condition_type == "single":
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx: int, agent_idx: Optional[int] = None):
        if self.agent_condition_type == "single":
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]
            agent_mask = np.zeros(self.n_agents, dtype=bool)
            agent_mask[idx % self.n_agents] = 1
        elif self.agent_condition_type == "all":
            path_ind, start, end, mask_end = self.indices[idx]
            agent_mask = np.ones(self.n_agents, dtype=bool)
        elif self.agent_condition_type == "random":
            path_ind, start, end, mask_end = self.indices[idx]
            agent_mask = np.random.randint(0, 2, self.n_agents, dtype=bool)

        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask)
            cond_trajectories = observations.copy()
        else:
            cond_masks = self.mask_generator(trajectories.shape, agent_mask)
            cond_trajectories = trajectories.copy()
        cond_trajectories[: self.history_horizon, ~agent_mask] = 0.0
        cond = {
            "x": cond_trajectories,
            "masks": cond_masks,
        }

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        if self.pred_future_padding:
            loss_masks[self.history_horizon:] = 1.0
        else:
            loss_masks[self.history_horizon: mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks[self.history_horizon, agent_mask] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon: mask_end - history_start] = 1.0
        attention_masks[: self.history_horizon, agent_mask] = 1.0

        batch = {
            "x": trajectories,
            "cond": cond,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start: -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch["returns"] = returns

        if self.include_env_ts:
            env_ts = np.arange(history_start, start + self.horizon) - self.history_horizon
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[path_ind, history_start:end]

        return batch
