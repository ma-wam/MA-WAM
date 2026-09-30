# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
MeanFlow Policy Implementation

Implements the MeanFlow algorithm from "Mean Flows for One-step Generative Modeling".
Uses average velocity instead of instantaneous velocity for flow-based policy learning.
"""

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from .mlp_meanflow import MeanFlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


def stopgrad(x):
    """Stop gradient computation"""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss as defined in MeanFlow paper.

    Args:
        error: Tensor of prediction errors (B, Ta, Da) for actions
        gamma: Power parameter for adaptive weighting (default: 0.5)
        c: Small constant for numerical stability (default: 1e-3)

    Returns:
        Scalar loss value
    """
    # Compute squared error per sample - flatten trajectory and action dimensions
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)), keepdim=False)
    p = 1.0 - gamma
    # Adaptive weight: w = 1 / (||Delta||^2 + c)^p
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    # Apply stopped gradient to weights
    return (stopgrad(w) * loss).mean()


class MeanFlow(nn.Module):
    """
    MeanFlow Policy implementing average velocity-based flow matching.

    This implementation follows the MeanFlow paper which introduces average velocity
    as an alternative to instantaneous velocity in flow matching methods.
    """

    def __init__(
        self,
        network: MeanFlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        # MeanFlow specific parameters
        flow_ratio: float = 0.5,  # Ratio of samples using flow consistency
        gamma: float = 0.5,      # Parameter for adaptive L2 loss
        c: float = 1e-3,         # Stability constant for adaptive loss
        sample_t_type: str = 'uniform',
        use_adaptive_loss: bool = False  # Whether to use adaptive L2 loss
    ):
        """
        Initialize MeanFlow model.

        Args:
            network: MeanFlowMLP network for average velocity prediction
            device: Device to run the model on
            horizon_steps: Number of steps in trajectory horizon
            action_dim: Dimension of action space
            act_min: Minimum action value for clipping
            act_max: Maximum action value for clipping
            obs_dim: Dimension of observation space
            max_denoising_steps: Maximum denoising steps (for compatibility)
            seed: Random seed for reproducibility
            flow_ratio: Ratio of samples using flow consistency constraint
            gamma: Adaptive loss parameter
            c: Stability constant for adaptive loss
            sample_t_type: Time sampling type ('uniform', 'beta', 'logitnormal')
        """
        super().__init__()

        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)

        # MeanFlow specific parameters
        self.flow_ratio = flow_ratio
        self.gamma = gamma
        self.c = c
        self.sample_t_type = sample_t_type
        self.use_adaptive_loss = use_adaptive_loss
        self.use_improved_meanflow = False  # Will be set by wrapper if needed


    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """
        Generate trajectory following official MeanFlow: xt = (1-t)*x1 + t*x0
        This goes from data (t=0) to noise (t=1), consistent with official implementation

        Args:
            x1: Target data tensor (B, Ta, Da)
            x0: Initial noise tensor (B, Ta, Da)
            t: Time tensor (B,)

        Returns:
            Interpolated trajectory xt
        """
        t_ = t.view(x1.shape[0], 1, 1).expand_as(x1)
        xt = (1 - t_) * x1 + t_ * x0  # From data to noise (official direction)
        return xt

    def loss(self, x1: Tensor, cond: dict) -> Tensor:
        """
        Compute MeanFlow loss following the official implementation.

        Uses the simplified MeanFlow identity with stable JVP computation.

        Args:
            x1: Real action trajectories (B, Ta, Da)
            cond: Condition dictionary with 'state' key

        Returns:
            Scalar loss tensor
        """
        batch_size = x1.shape[0]

        # Normalize action data to [-1, 1] range for better stability
        x1 = torch.clamp(x1, *self.act_range)

        # Sample time pairs (t, r) using lognormal distribution for stability
        mu, sigma = -0.4, 1.0  # Same as original MeanFlow
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair (ensures r <= t)
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Apply flow consistency for flow_ratio fraction of samples
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=self.device)
        r = torch.tensor(r_np, device=self.device)

        # Expand time dimensions
        t_ = t.view(-1, 1, 1)
        r_ = r.view(-1, 1, 1)

        # Sample noise and generate trajectory
        x0 = torch.randn_like(x1)

        # Generate trajectory: xt = (1-t)*x1 + t*x0 (from data to noise)
        xt = (1 - t_) * x1 + t_ * x0

        # Compute conditional velocity (from data to noise)
        v_cond = x0 - x1

        # Create partial function for network evaluation
        def network_fn(z, t_val, r_val):
            return self.network(z, t_val, r_val, cond)

        if self.use_improved_meanflow:
            # ========== Improved MeanFlow (iMF) - v-loss ==========
            # Reference: "Improved Mean Flows" (Geng et al., 2025) Algorithm 1

            # Step 1: Predict instantaneous velocity v using boundary condition u(z,t,t)
            v_pred = self.network(xt, t, t, cond)

            # Step 2: JVP with v_pred as tangent (not v_cond)
            u, dudt = torch.autograd.functional.jvp(
                network_fn,
                (xt, t, r),
                (v_pred, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False
            )

            # Step 3: Construct compound velocity V = u + (t-r) * du/dt
            V = u + (t_ - r_) * stopgrad(dudt)

            # Step 4: v-loss (regress V to conditional velocity)
            if hasattr(self, 'use_adaptive_loss') and self.use_adaptive_loss:
                error = V - v_cond
                return adaptive_l2_loss(error, gamma=self.gamma, c=self.c)
            else:
                return F.mse_loss(V, v_cond)
        else:
            # ========== Original MeanFlow - u-loss ==========
            # Use official JVP computation
            u, dudt = torch.autograd.functional.jvp(
                network_fn,
                (xt, t, r),
                (v_cond, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False
            )

            # Apply MeanFlow identity: u_target = v - (t-r) * du/dt
            u_target = v_cond - (t_ - r_) * dudt

            # Compute loss using adaptive L2 if specified, otherwise standard MSE
            if hasattr(self, 'use_adaptive_loss') and self.use_adaptive_loss:
                error = u - stopgrad(u_target)
                return adaptive_l2_loss(error, gamma=self.gamma, c=self.c)
            else:
                return F.mse_loss(u, stopgrad(u_target))

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int = 5,  # Follow official MeanFlow default of 5 steps
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """
        Sample trajectories using MeanFlow following official multi-step sampling.

        Args:
            cond: Condition dictionary with 'state' key
            inference_steps: Number of inference steps (default 5, following official implementation)
            record_intermediate: Whether to record intermediate steps
            clip_intermediate_actions: Whether to clip actions to valid range
            z: Initial noise (if None, sample from Gaussian)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]

        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)

        # Initial noise
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)

        # Follow official MeanFlow sampling logic
        # Create time schedule: from 1.0 (noise) to 0.0 (data)
        t_vals = torch.linspace(1.0, 0.0, inference_steps + 1, device=self.device)

        for i in range(inference_steps):
            # Current time values
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]

            # Create batch-wise time tensors
            t = torch.full((B,), t_curr, device=self.device)
            r = torch.full((B,), r_next, device=self.device)

            # Predict average velocity u(z_t, r, t)
            u = self.network(x_hat, t, r, cond)

            # Apply MeanFlow sampling formula: z_r = z_t - (t-r) * u
            # Reshape time difference for broadcasting
            time_diff = (t_curr - r_next)  # scalar difference
            x_hat = x_hat - time_diff * u

            # Optional clipping during intermediate steps
            if clip_intermediate_actions:
                x_hat = x_hat.clamp(*self.act_range)

            if record_intermediate:
                x_hat_list[i] = x_hat

        # Final clipping
        x_hat = x_hat.clamp(*self.act_range)

        return Sample(
            trajectories=x_hat,
            chains=x_hat_list if record_intermediate else None
        )