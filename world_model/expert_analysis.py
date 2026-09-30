"""
MoE 专家分析工具 - 分析 SoftMoE/SparseMoE 的专家路由和专门化模式

提供多种分析方法：
- 路由权重提取
- Agent-Expert 亲和度矩阵
- Agent 间交互模式 (通过共享 Expert Slot)
- 状态空间聚类 (按主导专家)
- 反事实专家消融
"""

import numpy as np
import torch
from collections import defaultdict


class MoEExpertAnalyzer:
    """
    MoE 世界模型的专家路由分析器。

    Args:
        world_model: 训练好的 MoEWorldModel
        device: 计算设备
    """

    def __init__(self, world_model, device='cuda'):
        self.world_model = world_model
        self.device = device
        self.world_model.eval()
        self.world_model.to(device)

    @torch.no_grad()
    def extract_routing_weights(self, obs, act):
        """
        提取一个 batch 的路由权重。

        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
        Returns:
            dict with dynamics and reward routing info
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) if not isinstance(obs, torch.Tensor) else obs.to(self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device) if not isinstance(act, torch.Tensor) else act.to(self.device)

        _, dyn_routing = self.world_model.dynamics.forward_with_routing(obs, act)

        # Reward routing 需要 next_obs
        next_obs = self.world_model.predict_next_obs(obs, act)
        _, _, rew_routing = self.world_model.reward_predictor.forward_with_routing(obs, act, next_obs)

        return {
            'dynamics': dyn_routing,
            'reward': rew_routing,
        }

    @torch.no_grad()
    def compute_agent_expert_affinity(self, obs_data, act_data, path_lengths=None,
                                       n_samples=5000):
        """
        计算 Agent-Expert 亲和度矩阵。

        对于 SoftMoE dynamics: 统计每个 agent 对各 expert slot 的平均 combine 权重。
        对于 SparseMoE reward: 统计每个 agent 被路由到各 expert 的频率。

        Args:
            obs_data: (N, n_agents, obs_dim) numpy array
            act_data: (N, n_agents, act_dim) numpy array
            path_lengths: 可选，episode 长度数组
            n_samples: 采样数量

        Returns:
            dict with:
                - dyn_affinity: (n_agents, n_experts) - dynamics expert 亲和度
                - rew_affinity: (n_agents, n_experts) - reward expert 亲和度
        """
        n_total = len(obs_data)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

        n_experts_dyn = self.world_model.dynamics.n_experts
        n_slots = self.world_model.dynamics.n_slots
        n_agents = self.world_model.dynamics.n_agents
        n_experts_rew = self.world_model.reward_predictor.n_experts

        # 统计 dispatch 和 combine 权重
        dyn_combine_sum = np.zeros((n_agents, n_experts_dyn))
        rew_route_count = np.zeros((n_agents, n_experts_rew))
        count = 0

        batch_size = 256
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            obs = torch.tensor(obs_data[batch_idx], dtype=torch.float32, device=self.device)
            act = torch.tensor(act_data[batch_idx], dtype=torch.float32, device=self.device)

            _, dyn_routing = self.world_model.dynamics.forward_with_routing(obs, act)
            next_obs = self.world_model.predict_next_obs(obs, act)
            _, _, rew_routing = self.world_model.reward_predictor.forward_with_routing(obs, act, next_obs)

            # Dynamics: combine_probs (batch, n_agents, total_slots)
            # 汇总到 expert 级别: 每个 expert 有 n_slots 个 slot
            combine = dyn_routing['combine_probs'].cpu().numpy()  # (B, n_agents, total_slots)
            for e in range(n_experts_dyn):
                slot_start = e * n_slots
                slot_end = slot_start + n_slots
                dyn_combine_sum[:, e] += combine[:, :, slot_start:slot_end].sum(axis=(0, 2))

            # Reward: top_k_indices (n_tokens, top_k)
            top_k = rew_routing['top_k_indices'].cpu().numpy()  # (B*n_agents, top_k)
            top_k = top_k.reshape(len(batch_idx), n_agents, -1)
            for a in range(n_agents):
                for expert_idx in top_k[:, a].flatten():
                    rew_route_count[a, expert_idx] += 1

            count += len(batch_idx)

        # 归一化
        dyn_affinity = dyn_combine_sum / count
        rew_affinity = rew_route_count / rew_route_count.sum(axis=1, keepdims=True)

        return {
            'dyn_affinity': dyn_affinity,
            'rew_affinity': rew_affinity,
        }

    @torch.no_grad()
    def compute_interaction_patterns(self, obs_data, act_data, n_samples=5000):
        """
        分析 Agent 间通过共享 Expert Slot 形成的隐式耦合。

        如果两个 agent 对同一个 slot 都有高 dispatch 权重，
        说明它们的信息在该 slot 中被混合，形成交互。

        Returns:
            interaction_matrix: (n_agents, n_agents) - 耦合强度
        """
        n_total = len(obs_data)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
        n_agents = self.world_model.dynamics.n_agents

        interaction_sum = np.zeros((n_agents, n_agents))
        count = 0

        batch_size = 256
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            obs = torch.tensor(obs_data[batch_idx], dtype=torch.float32, device=self.device)
            act = torch.tensor(act_data[batch_idx], dtype=torch.float32, device=self.device)

            _, dyn_routing = self.world_model.dynamics.forward_with_routing(obs, act)

            # dispatch_probs: (B, n_agents, total_slots)
            dp = dyn_routing['dispatch_probs'].cpu().numpy()

            # 耦合强度: coupling(i,j) = sum_s dp[:, i, s] * dp[:, j, s]
            # 即两个 agent 在各 slot 上 dispatch 权重的内积
            for b in range(len(batch_idx)):
                interaction_sum += dp[b] @ dp[b].T

            count += len(batch_idx)

        interaction_matrix = interaction_sum / count
        return interaction_matrix

    @torch.no_grad()
    def expert_specialization_by_state(self, obs_data, act_data, n_samples=5000):
        """
        按主导专家对状态进行聚类分析。

        对每个状态，找到 combine 权重最大的 expert，
        统计各 expert 处理的状态的均值和方差，分析空间分区。

        Returns:
            dict with:
                - dominant_expert: (n_samples, n_agents) - 每个样本每个 agent 的主导 expert
                - expert_state_stats: dict[expert_id] -> {mean, std, count}
        """
        n_total = len(obs_data)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

        n_experts = self.world_model.dynamics.n_experts
        n_slots = self.world_model.dynamics.n_slots
        n_agents = self.world_model.dynamics.n_agents

        all_dominant_experts = []
        all_obs = []

        batch_size = 256
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            obs = torch.tensor(obs_data[batch_idx], dtype=torch.float32, device=self.device)
            act = torch.tensor(act_data[batch_idx], dtype=torch.float32, device=self.device)

            _, dyn_routing = self.world_model.dynamics.forward_with_routing(obs, act)
            combine = dyn_routing['combine_probs'].cpu().numpy()  # (B, n_agents, total_slots)

            # 汇总到 expert 级别
            expert_weights = np.zeros((len(batch_idx), n_agents, n_experts))
            for e in range(n_experts):
                s, t = e * n_slots, (e + 1) * n_slots
                expert_weights[:, :, e] = combine[:, :, s:t].sum(axis=2)

            dominant = expert_weights.argmax(axis=2)  # (B, n_agents)
            all_dominant_experts.append(dominant)
            all_obs.append(obs_data[batch_idx])

        dominant_expert = np.concatenate(all_dominant_experts, axis=0)
        all_obs = np.concatenate(all_obs, axis=0)

        # 统计每个 expert 处理的状态
        expert_state_stats = {}
        for e in range(n_experts):
            # 找到 agent 0 被分配到 expert e 的样本 (以 agent 0 为代表)
            mask = dominant_expert[:, 0] == e
            if mask.sum() > 0:
                states = all_obs[mask, 0]  # (count, obs_dim)
                expert_state_stats[e] = {
                    'mean': states.mean(axis=0),
                    'std': states.std(axis=0),
                    'count': int(mask.sum()),
                }
            else:
                expert_state_stats[e] = {'mean': None, 'std': None, 'count': 0}

        return {
            'dominant_expert': dominant_expert,
            'expert_state_stats': expert_state_stats,
            'observations': all_obs,
        }

    @torch.no_grad()
    def counterfactual_expert_ablation(self, obs_data, act_data, next_obs_data,
                                        n_samples=1000):
        """
        反事实专家消融: 逐个屏蔽专家，观察预测精度下降。

        通过将某个专家的输出置零，衡量该专家对整体预测的贡献。

        Returns:
            dict with:
                - baseline_mse: 无消融时的 MSE
                - ablation_mse: (n_experts,) - 消融各专家后的 MSE
                - degradation: (n_experts,) - MSE 增量 (越大说明该专家越重要)
        """
        n_total = len(obs_data)
        indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

        obs = torch.tensor(obs_data[indices], dtype=torch.float32, device=self.device)
        act = torch.tensor(act_data[indices], dtype=torch.float32, device=self.device)
        target = torch.tensor(next_obs_data[indices], dtype=torch.float32, device=self.device)

        n_experts = self.world_model.dynamics.n_experts
        n_slots = self.world_model.dynamics.n_slots

        # Baseline prediction
        pred_baseline = self.world_model.predict_next_obs(obs, act)
        baseline_mse = torch.nn.functional.mse_loss(pred_baseline, target).item()

        # 消融各专家
        dynamics = self.world_model.dynamics
        tokens = torch.cat([obs, act], dim=-1)
        tokens = dynamics.input_norm(tokens)

        dispatch_logits = torch.matmul(tokens, dynamics.dispatch_weights)
        dispatch_probs = torch.nn.functional.softmax(dispatch_logits, dim=1)
        slot_inputs = torch.bmm(dispatch_probs.transpose(1, 2), tokens)

        # 计算所有专家输出
        all_expert_outputs = []
        for i, expert in enumerate(dynamics.experts):
            start = i * n_slots
            end = start + n_slots
            expert_input = slot_inputs[:, start:end]
            b, s, d = expert_input.shape
            expert_output = expert(expert_input.reshape(b * s, d))
            all_expert_outputs.append(expert_output.reshape(b, s, -1))

        combine_logits = torch.matmul(tokens, dynamics.combine_weights)
        combine_probs = torch.nn.functional.softmax(combine_logits, dim=-1)

        ablation_mse = np.zeros(n_experts)
        for ablate_idx in range(n_experts):
            # 将被消融专家的输出置零
            slot_outputs = []
            for i, eo in enumerate(all_expert_outputs):
                if i == ablate_idx:
                    slot_outputs.append(torch.zeros_like(eo))
                else:
                    slot_outputs.append(eo)
            all_slots = torch.cat(slot_outputs, dim=1)
            output = torch.bmm(combine_probs, all_slots)
            output = dynamics.output_norm(output)
            pred_ablated = obs + output
            ablation_mse[ablate_idx] = torch.nn.functional.mse_loss(pred_ablated, target).item()

        degradation = ablation_mse - baseline_mse

        return {
            'baseline_mse': baseline_mse,
            'ablation_mse': ablation_mse,
            'degradation': degradation,
        }
