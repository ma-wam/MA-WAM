"""Direct H-step trajectory-return predictor used as a selector baseline.

Unlike a world model, this module does not predict intermediate observations.  It
maps the current joint observation and a complete joint action sequence directly
to the undiscounted team return over that sequence.
"""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryReturnPredictor(nn.Module):
    """Predict an H-step scalar team return from a joint action proposal.

    Args:
        n_agents: Number of agents in the joint observation/action tensors.
        obs_dim: Per-agent observation dimension.
        act_dim: Per-agent action feature dimension.  For discrete actions this
            is the number of actions and inputs are expected to be one-hot.
        horizon: Number of proposed actions scored at once.
        hidden_dim: Width of the MLP.
        n_layers: Number of hidden affine layers.
        dropout: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        horizon: int = 8,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(n_agents, obs_dim, act_dim, horizon, hidden_dim, n_layers) <= 0:
            raise ValueError("All model dimensions and n_layers must be positive")

        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)

        input_dim = self.n_agents * (
            self.obs_dim + self.horizon * self.act_dim
        )
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                ]
            )
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        # Stored in the checkpoint.  Training regresses standardized targets,
        # while forward() exposes returns in the original reward scale.
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def set_target_stats(self, mean: float, std: float) -> None:
        safe_std = max(float(std), 1e-6)
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(safe_std)

    def _features(
        self, observation: torch.Tensor, action_sequence: torch.Tensor
    ) -> torch.Tensor:
        if observation.ndim != 3:
            raise ValueError(
                f"observation must have shape (B,A,obs_dim), got {tuple(observation.shape)}"
            )
        if action_sequence.ndim != 4:
            raise ValueError(
                "action_sequence must have shape (B,H,A,act_dim), "
                f"got {tuple(action_sequence.shape)}"
            )
        batch = observation.shape[0]
        expected_obs = (batch, self.n_agents, self.obs_dim)
        expected_act = (batch, self.horizon, self.n_agents, self.act_dim)
        if tuple(observation.shape) != expected_obs:
            raise ValueError(
                f"observation shape mismatch: expected {expected_obs}, got {tuple(observation.shape)}"
            )
        if tuple(action_sequence.shape) != expected_act:
            raise ValueError(
                f"action_sequence shape mismatch: expected {expected_act}, "
                f"got {tuple(action_sequence.shape)}"
            )
        return torch.cat(
            [observation.reshape(batch, -1), action_sequence.reshape(batch, -1)],
            dim=-1,
        )

    def predict_normalized(
        self, observation: torch.Tensor, action_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Return standardized predictions with shape ``(B,)``."""
        return self.network(self._features(observation, action_sequence)).squeeze(-1)

    def forward(
        self, observation: torch.Tensor, action_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Return predictions in the original undiscounted-return scale."""
        normalized = self.predict_normalized(observation, action_sequence)
        return normalized * self.target_std + self.target_mean

    def loss(
        self,
        observation: torch.Tensor,
        action_sequence: torch.Tensor,
        target_return: torch.Tensor,
    ) -> torch.Tensor:
        target_return = target_return.reshape(-1)
        target = (target_return - self.target_mean) / self.target_std
        prediction = self.predict_normalized(observation, action_sequence)
        return torch.mean((prediction - target) ** 2)

    def model_config(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }
