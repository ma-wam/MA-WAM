"""Centralized behavior-FQE critic used as a candidate ranker.

The checkpoint format matches the critics trained for Paper 3.  Ranking is
deliberately one-step: every CoFlow candidate is scored with Q(o_t, a_t), using
the same current joint observation and only its first executable joint action.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


class CentralizedFQECritic(nn.Module):
    """MLP architecture shared by the Paper-3 FQE training scripts."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"critic expected flattened obs_dim={self.obs_dim}, got {obs.shape[-1]}"
            )
        if act.shape[-1] != self.act_dim:
            raise ValueError(
                f"critic expected flattened act_dim={self.act_dim}, got {act.shape[-1]}"
            )
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def _infer_hidden(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "net.0.weight"
    if key not in state_dict:
        raise KeyError(f"FQE checkpoint is missing {key!r}")
    return int(state_dict[key].shape[0])


def load_fqe_critic(
    checkpoint_path: str, device: torch.device
) -> Tuple[CentralizedFQECritic, Dict[str, Any], str]:
    """Load a raw-space Paper-3 FQE checkpoint with strict shape checks."""

    path = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint = torch.load(path, map_location=device)
    if "critic" not in checkpoint or "config" not in checkpoint:
        raise KeyError("expected a Paper-3 FQE checkpoint with 'critic' and 'config'")

    config = dict(checkpoint["config"])
    state_dict = checkpoint["critic"]
    obs_dim = int(config["obs_dim"])
    act_dim = int(config["act_dim"])
    hidden = int(config.get("hidden", _infer_hidden(state_dict)))
    critic = CentralizedFQECritic(obs_dim, act_dim, hidden=hidden)
    critic.load_state_dict(state_dict, strict=True)
    return critic.to(device).eval(), config, path


@torch.no_grad()
def q_rank_score(
    critic: CentralizedFQECritic,
    start_obs_raw,
    action_sequences,
    device: torch.device,
) -> torch.Tensor:
    """Score M candidates in E environments with Q(o_t, a_t^(m)).

    Args:
        start_obs_raw: ``(E, n_agents, obs_dim)`` raw environment observations.
        action_sequences: ``(M, E, H, n_agents, action_dim)``.  Continuous
            actions are raw and discrete actions are one-hot, matching FQE.

    Returns:
        A tensor of shape ``(M, E)``; larger is better.
    """

    actions = torch.as_tensor(action_sequences, dtype=torch.float32, device=device)
    observations = torch.as_tensor(start_obs_raw, dtype=torch.float32, device=device)
    if actions.ndim != 5:
        raise ValueError(f"expected action_sequences rank 5, got shape={tuple(actions.shape)}")
    if observations.ndim != 3:
        raise ValueError(f"expected start_obs_raw rank 3, got shape={tuple(observations.shape)}")

    m_candidates, n_envs = actions.shape[:2]
    if observations.shape[0] != n_envs:
        raise ValueError(
            f"environment count mismatch: observations={observations.shape[0]}, actions={n_envs}"
        )

    joint_obs = observations.reshape(n_envs, -1)
    joint_obs = joint_obs.unsqueeze(0).expand(m_candidates, -1, -1)
    first_joint_action = actions[:, :, 0].reshape(m_candidates, n_envs, -1)
    scores = critic(
        joint_obs.reshape(m_candidates * n_envs, -1),
        first_joint_action.reshape(m_candidates * n_envs, -1),
    )
    return scores.reshape(m_candidates, n_envs)
