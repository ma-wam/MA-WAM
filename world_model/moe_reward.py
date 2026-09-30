"""
SparseMoE 奖励预测器 - 预测多智能体奖励 (s_t, a_t, s_{t+1}) -> r_t

采用 Sparse Mixture-of-Experts 架构，使用 top-k 路由选择专家，
带负载均衡辅助损失防止专家坍缩。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoEReward(nn.Module):
    """
    SparseMoE 奖励预测器

    Args:
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        n_experts: 专家数量
        hidden_dim: 隐层维度
        top_k: 每个 token 激活的专家数
    """

    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        n_experts=4,
        hidden_dim=256,
        top_k=2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.input_dim = obs_dim * 2 + act_dim  # concat(obs, act, next_obs)
        self.n_experts = n_experts
        self.top_k = top_k

        # 路由器
        self.router = nn.Linear(self.input_dim, n_experts)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_experts)
        ])

        self.input_norm = nn.LayerNorm(self.input_dim)

    def forward(self, obs, act, next_obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
            next_obs: (batch, n_agents, obs_dim)
        Returns:
            reward_pred: (batch, n_agents, 1)
            aux_loss: 负载均衡损失 (标量)
        """
        # 构建输入 token
        tokens = torch.cat([obs, act, next_obs], dim=-1)  # (batch, n_agents, input_dim)
        tokens = self.input_norm(tokens)

        batch_size = tokens.shape[0]
        n_tokens = batch_size * self.n_agents
        flat_tokens = tokens.reshape(n_tokens, self.input_dim)

        # 路由: top-k 选择
        router_logits = self.router(flat_tokens)  # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # 归一化 top-k 权重
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 专家计算
        # 为简单高效起见，计算所有专家输出然后 gather
        all_expert_outputs = torch.stack([
            expert(flat_tokens) for expert in self.experts
        ], dim=1)  # (n_tokens, n_experts, 1)

        # Gather top-k 专家输出并加权求和
        selected_outputs = torch.gather(
            all_expert_outputs,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, 1),
        )  # (n_tokens, top_k, 1)

        reward_pred = (top_k_probs.unsqueeze(-1) * selected_outputs).sum(dim=1)
        reward_pred = reward_pred.reshape(batch_size, self.n_agents, 1)

        # 负载均衡辅助损失 (鼓励各专家被均匀使用)
        # f_i = 每个专家被选中的比例, P_i = 每个专家的平均路由概率
        expert_mask = F.one_hot(top_k_indices, self.n_experts).float()  # (n_tokens, top_k, n_experts)
        expert_usage = expert_mask.sum(dim=1).mean(dim=0)  # (n_experts,)
        expert_prob_mean = router_probs.mean(dim=0)  # (n_experts,)
        aux_loss = (expert_usage * expert_prob_mean).sum() * self.n_experts

        return reward_pred, aux_loss

    def forward_with_routing(self, obs, act, next_obs):
        """
        与 forward() 相同，但额外返回路由信息用于专家分析。

        Returns:
            reward_pred: (batch, n_agents, 1)
            aux_loss: 标量
            routing_info: dict containing:
                - router_probs: (n_tokens, n_experts) - 完整路由概率
                - top_k_indices: (n_tokens, top_k) - 被选中的专家索引
                - top_k_probs: (n_tokens, top_k) - 被选中专家的权重
        """
        tokens = torch.cat([obs, act, next_obs], dim=-1)
        tokens = self.input_norm(tokens)

        batch_size = tokens.shape[0]
        n_tokens = batch_size * self.n_agents
        flat_tokens = tokens.reshape(n_tokens, self.input_dim)

        router_logits = self.router(flat_tokens)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs_normed = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        all_expert_outputs = torch.stack([
            expert(flat_tokens) for expert in self.experts
        ], dim=1)

        selected_outputs = torch.gather(
            all_expert_outputs, dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, 1),
        )

        reward_pred = (top_k_probs_normed.unsqueeze(-1) * selected_outputs).sum(dim=1)
        reward_pred = reward_pred.reshape(batch_size, self.n_agents, 1)

        expert_mask = F.one_hot(top_k_indices, self.n_experts).float()
        expert_usage = expert_mask.sum(dim=1).mean(dim=0)
        expert_prob_mean = router_probs.mean(dim=0)
        aux_loss = (expert_usage * expert_prob_mean).sum() * self.n_experts

        routing_info = {
            'router_probs': router_probs,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
        }
        return reward_pred, aux_loss, routing_info
