import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_dynamics import MonolithicDynamics, IndependentDynamics


class MonolithicReward(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=512, n_layers=2):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        input_dim = n_agents * (obs_dim * 2 + act_dim)
        output_dim = n_agents

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_norm = nn.LayerNorm(input_dim)

    def forward(self, obs, act, next_obs):
        batch_size = obs.shape[0]
        x = torch.cat([obs, act, next_obs], dim=-1).reshape(batch_size, -1)
        x = self.input_norm(x)
        rew = self.net(x).reshape(batch_size, self.n_agents, 1)
        return rew


class IndependentReward(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=512, n_layers=2):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        input_dim = obs_dim * 2 + act_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.input_norm = nn.LayerNorm(input_dim)

    def forward(self, obs, act, next_obs):
        batch_size = obs.shape[0]
        x = torch.cat([obs, act, next_obs], dim=-1)
        x = self.input_norm(x)
        flat = x.reshape(batch_size * self.n_agents, -1)
        rew = self.net(flat).reshape(batch_size, self.n_agents, 1)
        return rew


class BaselineWorldModel(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=512, model_type="monolithic"):
        super().__init__()
        if model_type not in {"monolithic", "independent"}:
            raise ValueError(f"unknown model_type: {model_type}")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        if model_type == "monolithic":
            self.dynamics = MonolithicDynamics(n_agents, obs_dim, act_dim, hidden_dim=hidden_dim)
            self.reward_predictor = MonolithicReward(n_agents, obs_dim, act_dim, hidden_dim=hidden_dim)
        else:
            self.dynamics = IndependentDynamics(n_agents, obs_dim, act_dim, hidden_dim=hidden_dim)
            self.reward_predictor = IndependentReward(n_agents, obs_dim, act_dim, hidden_dim=hidden_dim)

    def forward(self, obs, act):
        next_obs_pred = self.dynamics(obs, act)
        reward_pred = self.reward_predictor(obs, act, next_obs_pred.detach())
        return next_obs_pred, reward_pred

    def loss(self, obs, act, next_obs_target, reward_target):
        if reward_target.dim() == 2:
            reward_target = reward_target.unsqueeze(-1)
        next_obs_pred = self.dynamics(obs, act)
        reward_pred = self.reward_predictor(obs, act, next_obs_pred.detach())
        dyn_loss = F.mse_loss(next_obs_pred, next_obs_target)
        rew_loss = F.mse_loss(reward_pred, reward_target)
        total_loss = dyn_loss + rew_loss
        return total_loss, {
            "dynamics_loss": dyn_loss.item(),
            "reward_loss": rew_loss.item(),
            "total_loss": total_loss.item(),
        }

    def predict_next_obs(self, obs, act):
        return self.dynamics(obs, act)

    def predict_reward(self, obs, act, next_obs):
        return self.reward_predictor(obs, act, next_obs)
