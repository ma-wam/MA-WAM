"""
合成轨迹生成器 - 使用训练好的世界模型自回归生成合成 episode

生成的数据格式与 M2Flow 的离线数据完全一致 (obs.npy, actions.npy, rewards.npy, path_lengths.npy)，
可直接用于 M2Flow 训练。
"""

import os
import numpy as np
import torch


class SyntheticRolloutGenerator:
    """
    使用世界模型生成合成轨迹

    策略: 从真实数据中采样初始状态和动作序列，
    用世界模型自回归预测状态转移，生成合成 episode。

    Args:
        world_model: 训练好的 MoEWorldModel
        real_data_path: 真实数据目录 (包含 obs.npy, actions.npy 等)
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        rollout_horizon: 每条合成轨迹的长度
        n_rollout_episodes: 生成的 episode 数量
        action_noise_std: 动作噪声标准差 (增加多样性)
    """

    def __init__(
        self,
        world_model,
        real_data_path,
        n_agents,
        obs_dim,
        act_dim,
        rollout_horizon=50,
        n_rollout_episodes=1000,
        action_noise_std=0.1,
        discrete_action=False,
        num_actions=None,
        teacher_force_reward=False,
        return_filter_min=None,
        return_filter_max=None,
        return_bins=None,
        return_bin_weights=None,
        start_mode="random",
        seed=None,
    ):
        self.world_model = world_model
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rollout_horizon = rollout_horizon
        self.n_rollout_episodes = n_rollout_episodes
        self.action_noise_std = action_noise_std
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        # If True, copy rewards from the sampled real segment instead of
        # using the WM reward head. Avoids the systematic reward-head bias
        # observed in quality audits (synth return != real return sign/magnitude).
        self.teacher_force_reward = teacher_force_reward
        self.return_filter_min = return_filter_min
        self.return_filter_max = return_filter_max
        self.return_bins = return_bins
        self.return_bin_weights = return_bin_weights
        self.start_mode = start_mode
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        # 加载真实数据
        self.real_obs = np.load(os.path.join(real_data_path, "obs.npy"))
        self.real_actions = np.load(os.path.join(real_data_path, "actions.npy"))
        self.real_rewards = np.load(os.path.join(real_data_path, "rewards.npy"))
        self.real_path_lengths = np.load(os.path.join(real_data_path, "path_lengths.npy"))

        # SMAC 有 legals.npy; MAMuJoCo 没有 —— 可选加载
        legals_file = os.path.join(real_data_path, "legals.npy")
        self.real_legals = np.load(legals_file) if os.path.exists(legals_file) else None

        # 构建 episode 起始索引
        self.episode_starts = np.zeros(len(self.real_path_lengths), dtype=int)
        self.episode_starts[1:] = np.cumsum(self.real_path_lengths[:-1])
        self.episode_returns = self._compute_episode_returns()
        self.eligible_episode_indices = self._build_eligible_episode_indices()
        self._return_bin_episode_indices = None
        self._return_bin_probs = None
        if self.return_bins is not None:
            self._build_return_biased_episode_sampler()

    @torch.no_grad()
    def generate(self, output_path, device="cuda", batch_size=64):
        """
        生成合成轨迹并保存为 npy 文件

        Args:
            output_path: 输出目录
            device: 计算设备
            batch_size: 批量 rollout 大小
        """
        os.makedirs(output_path, exist_ok=True)
        self.world_model.eval()

        all_obs = []
        all_actions = []
        all_rewards = []
        all_legals = [] if self.real_legals is not None else None
        path_lengths = []
        # quality score per trajectory: mean L2 distance between WM-predicted obs and
        # real obs at the same (sampled) start point. Lower is better — the WM stayed
        # closer to the actual real-world transition.
        all_quality = []

        n_generated = 0
        while n_generated < self.n_rollout_episodes:
            cur_batch = min(batch_size, self.n_rollout_episodes - n_generated)

            # 采样初始状态、动作序列、真实 reward、real_obs 序列、(可选)legals
            sampled = self._sample_real_segments(cur_batch)
            init_obs, action_seqs_raw, real_rew_seq, real_obs_seq, legals_seq = sampled
            init_obs = torch.tensor(init_obs, dtype=torch.float32, device=device)
            # real_obs_seq: (batch, horizon, n_agents, obs_dim) — used for quality scoring
            real_obs_tensor = torch.tensor(real_obs_seq, dtype=torch.float32, device=device)

            if self.discrete_action and self.num_actions:
                # 离散动作: 随机替换部分动作增加多样性，然后 one-hot
                act_int = action_seqs_raw.astype(int)  # (batch, horizon, n_agents)
                if self.action_noise_std > 0:
                    mask = self.rng.random(act_int.shape) < self.action_noise_std
                    random_acts = self.rng.randint(0, self.num_actions, act_int.shape)
                    act_int = np.where(mask, random_acts, act_int)
                # one-hot encode
                B, H, A = act_int.shape
                one_hot = np.zeros((B, H, A, self.num_actions), dtype=np.float32)
                for a in range(A):
                    one_hot[np.arange(B)[:, None], np.arange(H)[None, :], a, act_int[:, :, a]] = 1.0
                action_seqs = torch.tensor(one_hot, dtype=torch.float32, device=device)
            else:
                action_seqs = torch.tensor(action_seqs_raw, dtype=torch.float32, device=device)
                # 添加连续动作噪声
                if self.action_noise_std > 0:
                    noise = torch.randn_like(action_seqs) * self.action_noise_std
                    action_seqs = action_seqs + noise

            # 自回归 rollout
            obs_traj = [init_obs]
            reward_traj = []
            current_obs = init_obs

            if self.teacher_force_reward:
                # real_rew_seq shape: (batch, horizon, n_agents) or (batch, horizon, n_agents, 1)
                real_rew_tensor = torch.tensor(real_rew_seq, dtype=torch.float32, device=device)
                if real_rew_tensor.ndim == 3:
                    real_rew_tensor = real_rew_tensor.unsqueeze(-1)  # (B,H,n_agents,1)

            for t in range(self.rollout_horizon - 1):
                act_t = action_seqs[:, t]  # (batch, n_agents, act_dim)
                next_obs = self.world_model.predict_next_obs(current_obs, act_t)
                if self.teacher_force_reward:
                    reward = real_rew_tensor[:, t]  # (batch, n_agents, 1)
                else:
                    reward = self.world_model.predict_reward(current_obs, act_t, next_obs)

                obs_traj.append(next_obs)
                reward_traj.append(reward)
                current_obs = next_obs

            # 最后一步的 reward
            if len(reward_traj) < self.rollout_horizon:
                if self.teacher_force_reward:
                    reward_traj.append(real_rew_tensor[:, self.rollout_horizon - 1])
                else:
                    # 补齐最后一步 reward
                    last_reward = torch.zeros(cur_batch, self.n_agents, 1, device=device)
                    reward_traj.append(last_reward)

            # 转换为 numpy
            obs_array = torch.stack(obs_traj, dim=1).cpu().numpy()  # (batch, horizon, n_agents, obs_dim)
            act_array = action_seqs[:, :self.rollout_horizon].cpu().numpy()  # (batch, horizon, n_agents, act_dim)
            rew_array = torch.cat(reward_traj, dim=-1).transpose(1, 2).cpu().numpy()  # 调整形状

            # 处理 reward 形状: (batch, horizon, n_agents, 1) -> (batch, horizon, n_agents, 1)
            rew_stacked = torch.stack(reward_traj, dim=1).cpu().numpy()  # (batch, horizon, n_agents, 1)

            # 计算每条轨迹的质量分数：WM 预测 obs 与真实 obs 的 L2 偏差均值
            # 用 numpy 计算更稳（不依赖 torch dim tuple 的版本支持）
            real_obs_np = real_obs_tensor.cpu().numpy()
            sq = (obs_array - real_obs_np) ** 2  # (batch, horizon, n_agents, obs_dim)
            traj_score = sq.reshape(sq.shape[0], -1).mean(axis=1)  # (batch,)

            for i in range(cur_batch):
                ep_obs = obs_array[i]  # (horizon, n_agents, obs_dim)
                ep_act = act_array[i]  # (horizon, n_agents, act_dim)
                ep_rew = rew_stacked[i]  # (horizon, n_agents, 1)

                all_obs.append(ep_obs)
                all_actions.append(ep_act)
                all_rewards.append(ep_rew)
                if all_legals is not None:
                    all_legals.append(legals_seq[i])
                all_quality.append(traj_score[i])
                path_lengths.append(self.rollout_horizon)

            n_generated += cur_batch
            print(f"  Generated {n_generated}/{self.n_rollout_episodes} episodes")

        # 拼接
        all_obs = np.concatenate(all_obs, axis=0)        # (T, n_agents, obs_dim)
        all_actions = np.concatenate(all_actions, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        path_lengths = np.array(path_lengths)

        # ---- 重要: 保存格式必须与真实数据集一致, 否则 ReplayBuffer.add_path
        # 在 buffer slot 形状不兼容时会触发 ValueError, 被 _inject_synthetic 的
        # try/except 静默吞掉, 导致合成 episode 被全部丢弃.
        # SMAC 真实数据: actions=(T, n_agents) int64,  rewards=(T, n_agents) float32
        # MAMuJoCo 真实数据: actions=(T, n_agents, act_dim) float32, rewards=(T, n_agents) float32
        if self.discrete_action and self.num_actions:
            # one-hot -> 整数索引
            actions_to_save = np.argmax(all_actions, axis=-1).astype(np.int64)
        else:
            actions_to_save = all_actions.astype(np.float32)

        # rewards: 去掉末尾的 (..,1) 维度，与真实一致
        rewards_to_save = all_rewards.astype(np.float32)
        if rewards_to_save.ndim >= 2 and rewards_to_save.shape[-1] == 1:
            rewards_to_save = rewards_to_save.squeeze(-1)

        np.save(os.path.join(output_path, "obs.npy"), all_obs.astype(np.float32))
        np.save(os.path.join(output_path, "actions.npy"), actions_to_save)
        np.save(os.path.join(output_path, "rewards.npy"), rewards_to_save)
        np.save(os.path.join(output_path, "path_lengths.npy"), path_lengths)
        if all_legals is not None:
            legals_arr = np.concatenate(all_legals, axis=0).astype(np.float32)
            np.save(os.path.join(output_path, "legals.npy"), legals_arr)
        # 每条 episode 的质量分数 (越小越接近真实 rollout)
        quality_arr = np.array(all_quality, dtype=np.float32)
        np.save(os.path.join(output_path, "quality_score.npy"), quality_arr)

        print(f"Synthetic data saved to {output_path}")
        print(f"  obs: {all_obs.shape} {all_obs.dtype}")
        print(f"  actions: {actions_to_save.shape} {actions_to_save.dtype}")
        print(f"  rewards: {rewards_to_save.shape} {rewards_to_save.dtype}")
        print(f"  episodes: {len(path_lengths)}")
        if all_legals is not None:
            print(f"  legals: {legals_arr.shape} {legals_arr.dtype}")
        print(f"  quality_score: {quality_arr.shape} mean={quality_arr.mean():.4f} "
              f"p10={np.percentile(quality_arr,10):.4f} "
              f"p50={np.percentile(quality_arr,50):.4f} "
              f"p90={np.percentile(quality_arr,90):.4f}")

        return output_path

    def _compute_episode_returns(self):
        """Per-agent episode return, matching the plotting/evaluation convention."""
        returns = []
        for ep_start, ep_len in zip(self.episode_starts, self.real_path_lengths):
            rew = np.asarray(self.real_rewards[ep_start: ep_start + ep_len])
            if rew.ndim >= 2:
                returns.append(float(rew.sum() / max(1, rew.shape[1])))
            else:
                returns.append(float(rew.sum()))
        return np.asarray(returns, dtype=np.float32)

    def _build_eligible_episode_indices(self):
        eligible = np.arange(len(self.real_path_lengths), dtype=np.int64)
        if self.return_filter_min is not None:
            eligible = eligible[self.episode_returns[eligible] >= float(self.return_filter_min)]
        if self.return_filter_max is not None:
            eligible = eligible[self.episode_returns[eligible] <= float(self.return_filter_max)]
        if len(eligible) == 0:
            raise ValueError(
                "No real episodes match return_filter_min/return_filter_max "
                f"({self.return_filter_min}, {self.return_filter_max})"
            )
        qs = np.percentile(self.episode_returns[eligible], [5, 50, 95])
        print(
            "[SyntheticRolloutGenerator] Eligible real episodes: "
            f"{len(eligible)}/{len(self.real_path_lengths)} "
            f"return mean={self.episode_returns[eligible].mean():.3f} "
            f"p5={qs[0]:.3f} p50={qs[1]:.3f} p95={qs[2]:.3f}"
        )
        return eligible

    def _build_return_biased_episode_sampler(self):
        bins = np.asarray(self.return_bins, dtype=np.float32)
        if bins.ndim != 1 or len(bins) < 2:
            raise ValueError("return_bins must be a 1D list with at least two edges")
        bins = np.sort(bins)
        bins[0] = -np.inf if not np.isneginf(bins[0]) else bins[0]
        bins[-1] = np.inf if not np.isposinf(bins[-1]) else bins[-1]

        weights = self.return_bin_weights
        if weights is None:
            weights = np.ones(len(bins) - 1, dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
        if len(weights) != len(bins) - 1:
            raise ValueError("return_bin_weights must have len(return_bins) - 1 entries")

        ret = self.episode_returns
        bin_indices = []
        active_weights = []
        messages = []
        eligible_mask = np.zeros(len(ret), dtype=bool)
        eligible_mask[self.eligible_episode_indices] = True
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if i == len(bins) - 2:
                mask = eligible_mask & (ret >= lo) & (ret <= hi)
            else:
                mask = eligible_mask & (ret >= lo) & (ret < hi)
            idx = np.flatnonzero(mask)
            bin_indices.append(idx)
            active_weights.append(float(weights[i]) if len(idx) > 0 else 0.0)
            messages.append(f"[{lo:g},{hi:g}]: n={len(idx)} w={active_weights[-1]:.3g}")

        active_weights = np.asarray(active_weights, dtype=np.float64)
        if active_weights.sum() <= 0:
            raise ValueError("return_bins produced no non-empty episode buckets")
        self._return_bin_episode_indices = bin_indices
        self._return_bin_probs = active_weights / active_weights.sum()
        print("[SyntheticRolloutGenerator] Return-biased episode sampling enabled")
        print("  " + " | ".join(messages))
        print(f"  probs={np.array2string(self._return_bin_probs, precision=3)}")

    def _sample_episode_index(self):
        if self._return_bin_episode_indices is not None:
            bin_id = self.rng.choice(len(self._return_bin_episode_indices), p=self._return_bin_probs)
            return int(self.rng.choice(self._return_bin_episode_indices[bin_id]))
        return int(self.rng.choice(self.eligible_episode_indices))

    def _sample_real_segments(self, batch_size):
        """从真实数据中采样初始状态/动作序列/reward 序列/真实 obs 序列/(可选) legals 序列。
        返回: (init_obs, action_seqs, real_rew_seqs, real_obs_seqs, legals_seqs_or_None)
        real_obs_seqs 用作质量分数 ground truth。
        """
        init_obs_list = []
        action_seq_list = []
        reward_seq_list = []
        real_obs_list = []
        legal_seq_list = [] if self.real_legals is not None else None

        for _ in range(batch_size):
            # 随机选一个 episode
            ep_idx = self._sample_episode_index()
            ep_start = self.episode_starts[ep_idx]
            ep_len = self.real_path_lengths[ep_idx]

            # 随机选一个起始点 (确保有足够长度的动作序列)
            max_start = max(0, ep_len - self.rollout_horizon)
            if self.start_mode in ("episode_start", "start", "zero"):
                start = 0
            elif self.start_mode == "random":
                start = self.rng.randint(0, max_start + 1)
            else:
                raise ValueError(f"Unknown start_mode: {self.start_mode}")

            # 取初始观测
            init_obs = self.real_obs[ep_start + start]  # (n_agents, obs_dim)

            # 取整段真实 obs 序列 (用于评分对比), padding 用最后一步重复
            obs_end = min(start + self.rollout_horizon, ep_len)
            real_obs_seg = np.asarray(self.real_obs[ep_start + start: ep_start + obs_end])
            if len(real_obs_seg) < self.rollout_horizon:
                pad_len = self.rollout_horizon - len(real_obs_seg)
                real_obs_seg = np.concatenate([
                    real_obs_seg,
                    np.repeat(real_obs_seg[-1:], pad_len, axis=0),
                ], axis=0)

            # 取动作序列 (可能不够长，用最后一步 padding)
            act_end = min(start + self.rollout_horizon, ep_len)
            actions = self.real_actions[ep_start + start: ep_start + act_end]
            if len(actions) < self.rollout_horizon:
                pad_len = self.rollout_horizon - len(actions)
                actions = np.concatenate([
                    actions,
                    np.repeat(actions[-1:], pad_len, axis=0),
                ], axis=0)

            # 取对应的真实 reward 序列 (同样位置，同长度 padding)
            rewards = self.real_rewards[ep_start + start: ep_start + act_end]
            if len(rewards) < self.rollout_horizon:
                pad_len = self.rollout_horizon - len(rewards)
                # padding 用 0 (表示 episode 结束后无 reward)，而不是重复最后一步
                pad = np.zeros((pad_len,) + rewards.shape[1:], dtype=rewards.dtype)
                rewards = np.concatenate([rewards, pad], axis=0)

            init_obs_list.append(init_obs)
            action_seq_list.append(actions)
            reward_seq_list.append(rewards)
            real_obs_list.append(real_obs_seg)

            if legal_seq_list is not None:
                legals = self.real_legals[ep_start + start: ep_start + act_end]
                if len(legals) < self.rollout_horizon:
                    pad_len = self.rollout_horizon - len(legals)
                    legals = np.concatenate([
                        legals,
                        np.repeat(legals[-1:], pad_len, axis=0),
                    ], axis=0)
                legal_seq_list.append(legals)

        init_obs = np.stack(init_obs_list)  # (batch, n_agents, obs_dim)
        action_seqs = np.stack(action_seq_list)  # (batch, horizon, n_agents, act_dim)
        reward_seqs = np.stack(reward_seq_list)  # (batch, horizon, n_agents[, 1])
        real_obs_seqs = np.stack(real_obs_list)  # (batch, horizon, n_agents, obs_dim)

        legals_arr = np.stack(legal_seq_list) if legal_seq_list is not None else None
        return init_obs, action_seqs, reward_seqs, real_obs_seqs, legals_arr
