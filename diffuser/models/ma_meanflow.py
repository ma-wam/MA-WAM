"""
Multi-Agent MeanFlow implementation based on madiff framework
"""

import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Dict, Optional
from collections import namedtuple

from .meanflow.meanflow import MeanFlow
from .helpers import apply_conditioning

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class MAMeanFlow(nn.Module):
    """
    Multi-Agent MeanFlow model that extends the single-agent MeanFlow
    to handle multi-agent scenarios similar to madiff.
    """

    def __init__(
        self,
        meanflow_model: MeanFlow,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        use_inv_dyn: bool = True,
        discrete_action: bool = False,
        num_actions: int = 0,
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        action_weight: float = 1.0,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        returns_loss_guided: bool = False,
        loss_guidence_w: float = 0.1,
        train_only_inv: bool = False,
        share_inv: bool = True,
        joint_inv: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim

        # MeanFlow specific parameters
        self.meanflow_model = meanflow_model
        self.use_inv_dyn = use_inv_dyn
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised

        # Loss configuration
        self.action_weight = action_weight
        self.loss_discount = loss_discount
        self.loss_weights = loss_weights
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight

        # Conditioning
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.returns_loss_guided = returns_loss_guided
        self.loss_guidence_w = loss_guidence_w

        # Inverse dynamics
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv
        self.joint_inv = joint_inv

        if self.use_inv_dyn:
            self._setup_inverse_dynamics()

        # Loss weights setup
        if loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        else:
            self.loss_weights = torch.ones(self.horizon)

        # Set up action indices for multi-agent
        self.action_indices = []
        start_idx = self.observation_dim
        for i in range(self.n_agents):
            end_idx = start_idx + self.action_dim
            self.action_indices.append((start_idx, end_idx))
            start_idx = end_idx + self.observation_dim if i < self.n_agents - 1 else end_idx

    def _setup_inverse_dynamics(self):
        """Setup inverse dynamics models for multi-agent setting."""
        if self.joint_inv:
            # Single inverse dynamics model for all agents
            self.inv_model = nn.Sequential(
                nn.Linear(self.observation_dim * 2 * self.n_agents, self.hidden_dim),
                nn.Mish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Mish(),
                nn.Linear(self.hidden_dim, self.action_dim * self.n_agents),
            )
        elif self.share_inv:
            # Shared inverse dynamics model
            self.inv_model = nn.Sequential(
                nn.Linear(self.observation_dim * 2, self.hidden_dim),
                nn.Mish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Mish(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
        else:
            # Independent inverse dynamics models for each agent
            self.inv_models = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.observation_dim * 2, self.hidden_dim),
                    nn.Mish(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Mish(),
                    nn.Linear(self.hidden_dim, self.action_dim),
                )
                for _ in range(self.n_agents)
            ])

    def extract_agent_data(self, x, agent_idx):
        """Extract data for a specific agent from the multi-agent tensor."""
        # x shape: (batch, horizon, transition_dim * n_agents)
        batch_size, horizon, total_dim = x.shape

        # Calculate start and end indices for this agent's data
        agent_transition_dim = self.observation_dim + self.action_dim
        start_idx = agent_idx * agent_transition_dim
        end_idx = start_idx + agent_transition_dim

        return x[:, :, start_idx:end_idx]

    def extract_actions_from_trajectory(self, x):
        """Extract actions for all agents from trajectory."""
        batch_size, horizon, total_dim = x.shape
        actions = []

        for i in range(self.n_agents):
            agent_data = self.extract_agent_data(x, i)
            # Actions are after observations in each agent's data
            agent_actions = agent_data[:, :, self.observation_dim:]
            actions.append(agent_actions)

        return torch.stack(actions, dim=1)  # (batch, n_agents, horizon, action_dim)

    def reconstruct_trajectory(self, observations, actions):
        """Reconstruct trajectory from observations and actions."""
        batch_size, n_agents, horizon, obs_dim = observations.shape
        _, _, _, act_dim = actions.shape

        # Interleave observations and actions for each agent
        trajectory = []
        for i in range(n_agents):
            agent_obs = observations[:, i]  # (batch, horizon, obs_dim)
            agent_acts = actions[:, i]      # (batch, horizon, act_dim)
            agent_traj = torch.cat([agent_obs, agent_acts], dim=-1)
            trajectory.append(agent_traj)

        # Concatenate all agents
        trajectory = torch.cat(trajectory, dim=-1)  # (batch, horizon, total_dim)
        return trajectory

    def loss(self, x, cond):
        """
        Compute loss for multi-agent MeanFlow.

        Args:
            x: trajectory tensor (batch, horizon, transition_dim * n_agents)
            cond: conditioning information

        Returns:
            loss dictionary
        """
        batch_size = x.shape[0]

        # Extract actions for all agents
        all_actions = self.extract_actions_from_trajectory(x)  # (batch, n_agents, horizon, action_dim)

        losses = {}
        total_loss = 0

        # Compute MeanFlow loss for each agent
        for agent_idx in range(self.n_agents):
            agent_actions = all_actions[:, agent_idx]  # (batch, horizon, action_dim)

            # Prepare agent-specific conditioning
            agent_cond = self._prepare_agent_conditioning(cond, agent_idx)

            # Compute MeanFlow loss for this agent
            agent_loss = self.meanflow_model.loss(agent_actions, agent_cond)
            losses[f'meanflow_agent_{agent_idx}'] = agent_loss
            total_loss += agent_loss

        # Inverse dynamics loss
        if self.use_inv_dyn and not self.train_only_inv:
            inv_loss = self._compute_inverse_dynamics_loss(x, all_actions)
            losses['inv_dynamics'] = inv_loss
            total_loss += inv_loss

        losses['total'] = total_loss
        return losses

    def _prepare_agent_conditioning(self, cond, agent_idx):
        """Prepare conditioning for a specific agent."""
        agent_cond = {}

        if 'state' in cond:
            # Extract state for this agent
            state = cond['state']
            if state.dim() == 3:  # (batch, n_agents, obs_dim)
                agent_cond['state'] = state[:, agent_idx:agent_idx+1]  # Keep agent dimension
            else:  # Assume concatenated format
                start_idx = agent_idx * self.observation_dim
                end_idx = start_idx + self.observation_dim
                agent_cond['state'] = state[:, start_idx:end_idx].unsqueeze(1)

        # Copy other conditioning as-is
        for key, value in cond.items():
            if key != 'state':
                agent_cond[key] = value

        return agent_cond

    def _compute_inverse_dynamics_loss(self, trajectories, actions):
        """Compute inverse dynamics loss."""
        # Extract observations from trajectories
        observations = []
        for i in range(self.n_agents):
            agent_data = self.extract_agent_data(trajectories, i)
            agent_obs = agent_data[:, :, :self.observation_dim]
            observations.append(agent_obs)
        observations = torch.stack(observations, dim=1)  # (batch, n_agents, horizon, obs_dim)

        if self.joint_inv:
            # Joint inverse dynamics
            return self._joint_inverse_dynamics_loss(observations, actions)
        else:
            # Independent or shared inverse dynamics
            return self._independent_inverse_dynamics_loss(observations, actions)

    def _joint_inverse_dynamics_loss(self, observations, actions):
        """Compute joint inverse dynamics loss."""
        batch_size, n_agents, horizon, obs_dim = observations.shape

        # Prepare input: concatenate consecutive observations for all agents
        obs_t = observations[:, :, :-1].reshape(batch_size, horizon-1, -1)
        obs_tp1 = observations[:, :, 1:].reshape(batch_size, horizon-1, -1)
        inv_input = torch.cat([obs_t, obs_tp1], dim=-1)

        # True actions
        true_actions = actions[:, :, :-1].reshape(batch_size, horizon-1, -1)

        # Predict actions
        pred_actions = self.inv_model(inv_input)

        return F.mse_loss(pred_actions, true_actions)

    def _independent_inverse_dynamics_loss(self, observations, actions):
        """Compute independent inverse dynamics loss."""
        total_loss = 0

        for agent_idx in range(self.n_agents):
            agent_obs = observations[:, agent_idx]  # (batch, horizon, obs_dim)
            agent_actions = actions[:, agent_idx]   # (batch, horizon, action_dim)

            # Prepare input for this agent
            obs_t = agent_obs[:, :-1]
            obs_tp1 = agent_obs[:, 1:]
            inv_input = torch.cat([obs_t, obs_tp1], dim=-1)

            # True actions
            true_actions = agent_actions[:, :-1]

            # Predict actions
            if self.share_inv:
                pred_actions = self.inv_model(inv_input)
            else:
                pred_actions = self.inv_models[agent_idx](inv_input)

            agent_loss = F.mse_loss(pred_actions, true_actions)
            total_loss += agent_loss

        return total_loss / self.n_agents

    def conditional_sample(self, cond, **sample_kwargs):
        """
        Sample from the multi-agent MeanFlow model.

        Args:
            cond: conditioning information
            **sample_kwargs: additional sampling arguments

        Returns:
            Sample with trajectories and optional chains
        """
        batch_size = cond.get('state', list(cond.values())[0]).shape[0]

        # Sample actions for each agent
        all_agent_actions = []
        all_agent_chains = []

        for agent_idx in range(self.n_agents):
            agent_cond = self._prepare_agent_conditioning(cond, agent_idx)

            # Sample from MeanFlow for this agent
            agent_sample = self.meanflow_model.sample(agent_cond, **sample_kwargs)
            all_agent_actions.append(agent_sample.trajectories)

            if agent_sample.chains is not None:
                all_agent_chains.append(agent_sample.chains)

        # Combine actions from all agents
        actions = torch.stack(all_agent_actions, dim=1)  # (batch, n_agents, horizon, action_dim)

        # Create dummy observations (zeros) to construct full trajectories
        obs_shape = (batch_size, self.n_agents, self.horizon, self.observation_dim)
        observations = torch.zeros(obs_shape, device=actions.device)

        # Reconstruct full trajectories
        trajectories = self.reconstruct_trajectory(observations, actions)

        # Handle chains if available
        chains = None
        if all_agent_chains and all_agent_chains[0] is not None:
            # Combine chains from all agents
            chains = torch.stack(all_agent_chains, dim=2)  # (steps, batch, n_agents, horizon, action_dim)

        return Sample(trajectories=trajectories, chains=chains)

    def forward(self, cond, deterministic=False, **sample_kwargs):
        """Forward pass for sampling."""
        return self.conditional_sample(cond, **sample_kwargs)