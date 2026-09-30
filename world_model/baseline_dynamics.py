"""
非 MoE 基线动力学模型 - 用于与 SoftMoE 动力学模型对比

提供两种基线:
1. MonolithicDynamics: 单网络处理所有智能体 (全连接)
2. IndependentDynamics: 每个智能体独立预测 (无交互)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonolithicDynamics(nn.Module):
    """
    单网络动力学模型 (非 MoE 基线)

    将所有智能体的 (obs, act) 拼接后用单个 MLP 处理，
    参数量与 SoftMoEDynamics 相当。

    Args:
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐层维度
        n_layers: MLP 层数
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        input_dim = n_agents * (obs_dim + act_dim)
        output_dim = n_agents * obs_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(obs_dim)

    def forward(self, obs, act):
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
        Returns:
            next_obs_pred: (batch, n_agents, obs_dim)
        """
        batch_size = obs.shape[0]

        # 拼接所有智能体的 (obs, act)
        tokens = torch.cat([obs, act], dim=-1)  # (batch, n_agents, obs_dim+act_dim)
        flat_input = tokens.reshape(batch_size, -1)  # (batch, n_agents*(obs_dim+act_dim))
        flat_input = self.input_norm(flat_input)

        output = self.net(flat_input)  # (batch, n_agents*obs_dim)
        output = output.reshape(batch_size, self.n_agents, self.obs_dim)
        output = self.output_norm(output)

        # 残差连接
        next_obs_pred = obs + output
        return next_obs_pred


class IndependentDynamics(nn.Module):
    """
    独立动力学模型 (非 MoE 基线)

    每个智能体使用相同参数的 MLP 独立预测，不考虑其他智能体的信息。
    用于验证 MoE 的智能体间交互建模能力。

    Args:
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐层维度
        n_layers: MLP 层数
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        input_dim = obs_dim + act_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, obs_dim))

        self.net = nn.Sequential(*layers)
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(obs_dim)

    def forward(self, obs, act):
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
        Returns:
            next_obs_pred: (batch, n_agents, obs_dim)
        """
        tokens = torch.cat([obs, act], dim=-1)  # (batch, n_agents, input_dim)
        tokens = self.input_norm(tokens)

        # 共享参数处理每个智能体
        batch_size = obs.shape[0]
        flat_tokens = tokens.reshape(batch_size * self.n_agents, -1)
        output = self.net(flat_tokens)
        output = output.reshape(batch_size, self.n_agents, self.obs_dim)
        output = self.output_norm(output)

        next_obs_pred = obs + output
        return next_obs_pred
