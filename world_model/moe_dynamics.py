"""
SoftMoE 动力学模型 - 学习多智能体状态转移函数 (s_t, a_t) -> s_{t+1}

采用 Soft Mixture-of-Experts 架构，将每个智能体的 (obs, act) 视为 token，
通过软路由分配给多个专家处理，捕获智能体间交互。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """单个专家网络: 2层 MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SoftMoEDynamics(nn.Module):
    """
    SoftMoE 动力学模型 (参考 M3W 设计)

    与 SparseMoE 不同，SoftMoE 使用软分配权重，不丢弃任何 token，
    更适合多智能体场景中每个智能体的信息都很重要的情况。

    Args:
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        n_experts: 专家数量
        hidden_dim: 专家隐层维度
        n_slots: SoftMoE 槽位数 (每个专家处理的聚合 token 数)
    """

    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        n_experts=8,
        hidden_dim=256,
        n_slots=4,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.n_experts = n_experts
        self.n_slots = n_slots
        total_slots = n_experts * n_slots

        # Dispatch 参数: 将 token 映射到 slot 空间
        self.dispatch_weights = nn.Parameter(
            torch.randn(self.input_dim, total_slots) * 0.02
        )

        # 专家网络
        self.experts = nn.ModuleList([
            ExpertMLP(self.input_dim, hidden_dim, obs_dim)
            for _ in range(n_experts)
        ])

        # Combine 参数: 将 slot 输出映射回 token 空间
        self.combine_weights = nn.Parameter(
            torch.randn(self.input_dim, total_slots) * 0.02
        )

        # Layer norm for stability
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.output_norm = nn.LayerNorm(obs_dim)

    def forward(self, obs, act):
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
        Returns:
            next_obs_pred: (batch, n_agents, obs_dim) - 预测的下一步观测
        """
        batch_size = obs.shape[0]

        # 1. 构建 token: concat(obs, act) -> (batch, n_agents, input_dim)
        tokens = torch.cat([obs, act], dim=-1)
        tokens = self.input_norm(tokens)

        # 2. 计算 dispatch 权重: softmax over agents for each slot
        # logits: (batch, n_agents, total_slots)
        dispatch_logits = torch.matmul(tokens, self.dispatch_weights)
        # 对 agent 维度 softmax -> 每个 slot 如何聚合各 agent 信息
        dispatch_probs = F.softmax(dispatch_logits, dim=1)

        # 3. 聚合: slot_input = sum(dispatch_prob * token)
        # (batch, total_slots, input_dim) = (batch, total_slots, n_agents) @ (batch, n_agents, input_dim)
        slot_inputs = torch.bmm(dispatch_probs.transpose(1, 2), tokens)

        # 4. 专家处理: 每个专家处理 n_slots 个 slot
        slot_outputs = []
        for i, expert in enumerate(self.experts):
            start = i * self.n_slots
            end = start + self.n_slots
            expert_input = slot_inputs[:, start:end]  # (batch, n_slots, input_dim)
            b, s, d = expert_input.shape
            expert_output = expert(expert_input.reshape(b * s, d))
            slot_outputs.append(expert_output.reshape(b, s, -1))

        # (batch, total_slots, obs_dim)
        all_slot_outputs = torch.cat(slot_outputs, dim=1)

        # 5. 计算 combine 权重: softmax over slots for each agent
        combine_logits = torch.matmul(tokens, self.combine_weights)
        # 对 slot 维度 softmax -> 每个 agent 如何组合各 slot 的输出
        combine_probs = F.softmax(combine_logits, dim=-1)

        # 6. 散射: output = sum(combine_prob * slot_output)
        # (batch, n_agents, obs_dim) = (batch, n_agents, total_slots) @ (batch, total_slots, obs_dim)
        output = torch.bmm(combine_probs, all_slot_outputs)
        output = self.output_norm(output)

        # 7. 残差连接: 预测状态变化量
        next_obs_pred = obs + output

        return next_obs_pred

    def forward_with_routing(self, obs, act):
        """
        与 forward() 相同，但额外返回路由信息用于专家分析。

        Returns:
            next_obs_pred: (batch, n_agents, obs_dim)
            routing_info: dict containing:
                - dispatch_probs: (batch, n_agents, total_slots)
                - combine_probs: (batch, n_agents, total_slots)
                - per_expert_outputs: list of (batch, n_slots, obs_dim)
        """
        batch_size = obs.shape[0]
        tokens = torch.cat([obs, act], dim=-1)
        tokens = self.input_norm(tokens)

        dispatch_logits = torch.matmul(tokens, self.dispatch_weights)
        dispatch_probs = F.softmax(dispatch_logits, dim=1)
        slot_inputs = torch.bmm(dispatch_probs.transpose(1, 2), tokens)

        per_expert_outputs = []
        slot_outputs = []
        for i, expert in enumerate(self.experts):
            start = i * self.n_slots
            end = start + self.n_slots
            expert_input = slot_inputs[:, start:end]
            b, s, d = expert_input.shape
            expert_output = expert(expert_input.reshape(b * s, d))
            expert_output = expert_output.reshape(b, s, -1)
            per_expert_outputs.append(expert_output)
            slot_outputs.append(expert_output)

        all_slot_outputs = torch.cat(slot_outputs, dim=1)

        combine_logits = torch.matmul(tokens, self.combine_weights)
        combine_probs = F.softmax(combine_logits, dim=-1)

        output = torch.bmm(combine_probs, all_slot_outputs)
        output = self.output_norm(output)
        next_obs_pred = obs + output

        routing_info = {
            'dispatch_probs': dispatch_probs,
            'combine_probs': combine_probs,
            'per_expert_outputs': per_expert_outputs,
        }
        return next_obs_pred, routing_info
