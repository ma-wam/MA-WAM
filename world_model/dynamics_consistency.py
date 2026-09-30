"""
Trajectory Dynamics Consistency (TDC) Module

在 M2Flow MeanFlow 训练过程中，利用冻结的世界模型提供动力学一致性正则化。
核心思路：从 MeanFlow 的速度预测重构伪去噪轨迹，用世界模型检查相邻状态的动力学一致性。

梯度流: x1_hat → predicted (M2Flow backbone) 和 inv_model (inverse dynamics)
世界模型参数冻结，不接受梯度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsConsistencyModule(nn.Module):
    """
    使用冻结的世界模型计算轨迹动力学一致性 (TDC) 损失。

    对 M2Flow 生成的伪去噪轨迹，检查相邻状态转移是否符合世界模型的预测。

    Args:
        world_model: 预训练的 MoEWorldModel (将被冻结)
        n_agents: 智能体数量
        obs_dim: 观测维度
        act_dim: 动作维度
        horizon_subsample: 每隔 k 步检查一次一致性 (default 1 = 全部检查)
    """

    def __init__(self, world_model, n_agents, obs_dim, act_dim, horizon_subsample=1):
        super().__init__()
        self.world_model = world_model
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon_subsample = horizon_subsample

        # 冻结世界模型参数
        for p in self.world_model.parameters():
            p.requires_grad = False
        self.world_model.eval()

    @torch.no_grad()
    def _wm_predict_next_obs(self, obs, act):
        """用冻结的世界模型预测下一步观测 (不计算梯度)"""
        return self.world_model.predict_next_obs(obs, act)

    def compute_tdc_loss(self, trajectory_obs, inv_model, share_inv=True,
                         joint_inv=False, loss_masks=None):
        """
        计算轨迹动力学一致性损失。

        Args:
            trajectory_obs: [B, H, n_agents, obs_dim] - 伪去噪观测轨迹 (有梯度)
            inv_model: M2Flow 的 inverse dynamics 模型 (可训练)
            share_inv: 是否使用共享 inverse dynamics
            joint_inv: 是否使用联合 inverse dynamics
            loss_masks: [B, H, n_agents, 1] - 可选的损失掩码

        Returns:
            tdc_loss: 标量损失
            info: 包含各步一致性指标的字典
        """
        batch_size, horizon, n_agents, obs_dim = trajectory_obs.shape

        # 选择要检查的时间步对
        if self.horizon_subsample > 1:
            timesteps = list(range(0, horizon - 1, self.horizon_subsample))
        else:
            timesteps = list(range(horizon - 1))

        if len(timesteps) == 0:
            return torch.tensor(0.0, device=trajectory_obs.device), {}

        # 收集所有时间步的 obs 对
        obs_t = trajectory_obs[:, timesteps]           # [B, T, n_agents, obs_dim]
        obs_tp1 = trajectory_obs[:, [t + 1 for t in timesteps]]  # [B, T, n_agents, obs_dim]

        T = len(timesteps)

        # 用 inverse dynamics 预测动作
        # x_comb: [B, T, n_agents, 2*obs_dim]
        x_comb = torch.cat([obs_t, obs_tp1], dim=-1)

        if joint_inv:
            # [B*T, n_agents*2*obs_dim] -> [B*T, n_agents, act_dim]
            x_flat = x_comb.reshape(batch_size * T, -1)
            pred_actions = inv_model(x_flat).reshape(batch_size * T, n_agents, self.act_dim)
        elif share_inv:
            # [B*T, n_agents, 2*obs_dim] -> [B*T, n_agents, act_dim]
            x_flat = x_comb.reshape(batch_size * T, n_agents, 2 * obs_dim)
            pred_actions = inv_model(x_flat)
        else:
            # Independent: per-agent forward
            x_flat = x_comb.reshape(batch_size * T, n_agents, 2 * obs_dim)
            pred_actions = torch.stack([
                inv_model[i](x_flat[:, i]) for i in range(n_agents)
            ], dim=1)

        # 用世界模型预测下一步观测 (frozen, no grad through WM params)
        obs_t_flat = obs_t.reshape(batch_size * T, n_agents, obs_dim)
        with torch.no_grad():
            obs_tp1_wm = self._wm_predict_next_obs(obs_t_flat.detach(), pred_actions.detach())

        # TDC 损失: MSE between M2Flow 轨迹的 obs_{t+1} 和 WM 预测的 obs_{t+1}
        obs_tp1_flat = obs_tp1.reshape(batch_size * T, n_agents, obs_dim)
        tdc_loss = F.mse_loss(obs_tp1_flat, obs_tp1_wm.detach())

        # 统计信息
        with torch.no_grad():
            per_step_mse = F.mse_loss(
                obs_tp1_flat, obs_tp1_wm, reduction='none'
            ).mean(dim=(1, 2)).reshape(batch_size, T)

        info = {
            'tdc_loss': tdc_loss.item(),
            'tdc_mean_mse': per_step_mse.mean().item(),
            'tdc_max_mse': per_step_mse.max().item(),
        }

        return tdc_loss, info
