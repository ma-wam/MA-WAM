"""
MoE 世界模型 - 组合动力学模型和奖励预测器

提供统一的前向推理和损失计算接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_dynamics import SoftMoEDynamics
from .moe_reward import SparseMoEReward


class MoEWorldModel(nn.Module):
    """
    组合世界模型: SoftMoE 动力学 + SparseMoE 奖励

    Args:
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        n_dyn_experts: 动力学专家数
        n_rew_experts: 奖励专家数
        hidden_dim: 隐层维度
        n_slots: SoftMoE 槽位数
    """

    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        n_dyn_experts=8,
        n_rew_experts=4,
        hidden_dim=256,
        n_slots=4,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.aux_loss_weight = aux_loss_weight

        self.dynamics = SoftMoEDynamics(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_experts=n_dyn_experts,
            hidden_dim=hidden_dim,
            n_slots=n_slots,
        )

        self.reward_predictor = SparseMoEReward(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_experts=n_rew_experts,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs, act):
        """
        前向推理: 预测下一步观测和奖励

        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
        Returns:
            next_obs_pred: (batch, n_agents, obs_dim)
            reward_pred: (batch, n_agents, 1)
        """
        next_obs_pred = self.dynamics(obs, act)
        reward_pred, _ = self.reward_predictor(obs, act, next_obs_pred.detach())
        return next_obs_pred, reward_pred

    def loss(self, obs, act, next_obs_target, reward_target):
        """
        计算训练损失

        Args:
            obs: (batch, n_agents, obs_dim)
            act: (batch, n_agents, act_dim)
            next_obs_target: (batch, n_agents, obs_dim)
            reward_target: (batch, n_agents, 1) or (batch, n_agents)
        Returns:
            total_loss: 标量
            info: 各项损失的字典
        """
        if reward_target.dim() == 2:
            reward_target = reward_target.unsqueeze(-1)

        # 动力学损失
        next_obs_pred = self.dynamics(obs, act)
        dyn_loss = F.mse_loss(next_obs_pred, next_obs_target)

        # 奖励损失 (用 detach 的预测状态，避免梯度干扰动力学)
        reward_pred, aux_loss = self.reward_predictor(
            obs, act, next_obs_pred.detach()
        )
        rew_loss = F.mse_loss(reward_pred, reward_target)

        total_loss = dyn_loss + rew_loss + self.aux_loss_weight * aux_loss

        info = {
            "dynamics_loss": dyn_loss.item(),
            "reward_loss": rew_loss.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, info

    def predict_next_obs(self, obs, act):
        """仅预测下一步观测 (用于 rollout)"""
        return self.dynamics(obs, act)

    def predict_reward(self, obs, act, next_obs):
        """仅预测奖励"""
        reward_pred, _ = self.reward_predictor(obs, act, next_obs)
        return reward_pred
