"""
Multi-Agent MeanFlow wrapper that integrates with madiff training framework
"""

import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import namedtuple

import diffuser.utils as utils
from .helpers import apply_conditioning
from .meanflow.dispersive_loss import compute_dispersive_loss, flatten_representation

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class MAMeanFlowWrapper(nn.Module):
    """
    Wrapper for MAMeanFlow that provides the same interface as GaussianDiffusion
    for compatibility with the madiff training framework.
    """

    def __init__(
        self,
        model,  # This will be the network architecture (e.g., TemporalUnet)
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
        predict_epsilon: bool = False,  # MeanFlow doesn't predict epsilon
        action_weight: float = 1.0,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        condition_dropout: Optional[float] = None,
        returns_loss_guided: bool = False,
        loss_guidence_w: float = 0.1,
        value_diffusion_model: nn.Module = None,
        train_only_inv: bool = False,
        share_inv: bool = True,
        joint_inv: bool = False,
        # MeanFlow specific parameters
        flow_ratio: float = 0.5,
        gamma: float = 0.5,
        c: float = 1e-3,
        use_adaptive_loss: bool = False,
        max_denoising_steps: int = 5,
        act_min: float = -1.0,
        act_max: float = 1.0,
        inv_loss_weight: float = 0.5,  # Weight for inverse dynamics loss
        # Dispersive regularisation
        use_dispersive_loss: bool = False,
        dispersive_loss_weight: float = 0.5,
        dispersive_loss_temperature: float = 0.5,
        dispersive_loss_type: str = "infonce_l2",
        dispersive_loss_layer: str = "output",
        # Improved MeanFlow (iMF) support
        use_improved_meanflow: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim

        # Use the passed model (SharedConvAttentionDeconv) instead of creating individual models
        self.model = model

        # MeanFlow specific parameters
        self.flow_ratio = flow_ratio
        self.gamma = gamma
        self.c = c
        self.use_adaptive_loss = use_adaptive_loss
        self.max_denoising_steps = max_denoising_steps
        self.act_min = act_min
        self.act_max = act_max
        self.inv_loss_weight = inv_loss_weight

        # Store parameters for compatibility
        self.use_inv_dyn = use_inv_dyn
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.n_timesteps = max_denoising_steps  # Use max_denoising_steps as n_timesteps
        self.clip_denoised = clip_denoised
        self.action_weight = action_weight
        self.loss_discount = loss_discount
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        if condition_dropout is None:
            # The canonical SharedConvAttentionDeconv stores this on its
            # underlying TemporalUnet.  Resolve it here so older serialized
            # diffusion configs (which did not pass this argument) keep
            # working and still use the model's configured probability.
            condition_dropout = getattr(model, "condition_dropout", None)
            if condition_dropout is None:
                condition_dropout = getattr(getattr(model, "net", None), "condition_dropout", None)
            if condition_dropout is None:
                nets = getattr(model, "nets", None)
                if nets is not None and len(nets) > 0:
                    condition_dropout = getattr(nets[0], "condition_dropout", None)
            if condition_dropout is None:
                condition_dropout = 0.0
        if not 0.0 <= float(condition_dropout) <= 1.0:
            raise ValueError("condition_dropout must lie in [0, 1]")
        self.condition_dropout = float(condition_dropout)
        self.returns_loss_guided = returns_loss_guided
        self.loss_guidence_w = loss_guidence_w
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv
        self.joint_inv = joint_inv

        # Dispersive loss config
        self.use_dispersive_loss = use_dispersive_loss
        self.dispersive_loss_weight = dispersive_loss_weight
        self.dispersive_loss_temperature = dispersive_loss_temperature
        self.dispersive_loss_type = dispersive_loss_type
        self.dispersive_loss_layer = dispersive_loss_layer

        # Improved MeanFlow (iMF) config
        self.use_improved_meanflow = use_improved_meanflow

        # World Model Guidance (TDC loss) - set externally via train script
        self.use_wm_guidance = False
        self.wm_guidance_module = None
        self.wm_guidance_weight = 0.1

        # Loss weights
        if loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        else:
            self.loss_weights = torch.ones(self.horizon)

        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight

        # Setup inverse dynamics if needed (same as GaussianDiffusion)
        if self.use_inv_dyn:
            self.inv_model = self._build_inv_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        """Build inverse dynamics model (same as GaussianDiffusion)."""
        if self.joint_inv:
            print("\n USE JOINT INV \n")
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            print("\n USE SHARED INV \n")
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            print("\n USE INDEPENDENT INV \n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        apply_cfg: bool = True,
        condition_dropout_mask: Optional[torch.Tensor] = None,
    ):
        """Get a velocity prediction.

        ``apply_cfg=True`` is the inference path and evaluates the conditional
        and unconditional branches.  Training passes ``apply_cfg=False`` and
        a pre-sampled keep mask, so it performs ordinary conditional dropout
        instead of optimizing a CFG-combined prediction.
        """
        if self.returns_condition:
            if apply_cfg:
                # Classifier-free guidance is used only for sampling.
                pred_cond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    use_dropout=False,
                )
                pred_uncond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    use_dropout=False,
                    force_dropout=True,
                )
                pred = pred_uncond + self.condition_guidance_w * (pred_cond - pred_uncond)
            else:
                pred = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    use_dropout=condition_dropout_mask is None,
                    force_dropout=False,
                    condition_dropout_mask=condition_dropout_mask,
                )
        else:
            pred = self.model(
                x, t, env_timestep=env_ts, attention_masks=attention_masks
            )

        return pred

    def _discretize_time(self, continuous_time: torch.Tensor) -> torch.Tensor:
        """Map t in [0, 1] to right-closed discrete bins.

        The right-closed convention maps t=1 to N-1 and, unlike floor(t*N),
        does not duplicate the highest bin at the next boundary.
        """
        return (
            torch.ceil(continuous_time * self.n_timesteps) - 1
        ).long().clamp(0, self.n_timesteps - 1)

    def _sampling_schedule(
        self, inference_steps: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Euler evaluation times and their exact integer bin ids."""
        inference_steps = int(inference_steps)
        if not 1 <= inference_steps <= self.n_timesteps:
            raise ValueError(
                "inference_steps must be between 1 and n_timesteps "
                f"({self.n_timesteps}), got {inference_steps}"
            )

        timesteps = torch.linspace(
            1.0, 0.0, inference_steps + 1, device=device
        )[:-1]
        # For t_i=(K-i)/K, ceil(t_i*N)-1 can be evaluated exactly with
        # integers as (N*(K-i)-1)//K, avoiding floating-point boundary error.
        remaining = torch.arange(
            inference_steps, 0, -1, device=device, dtype=torch.long
        )
        discrete_timesteps = torch.div(
            self.n_timesteps * remaining - 1,
            inference_steps,
            rounding_mode="floor",
        )
        return timesteps, discrete_timesteps

    def loss(self, x, cond, loss_masks=None, attention_masks=None, returns=None, env_ts=None, **kwargs):
        """
        Compute loss using MeanFlow approach with the shared model.
        Returns the same format as GaussianDiffusion for compatibility.

        Supports both original MeanFlow (u-loss) and Improved MeanFlow (v-loss).
        Reference: "Improved Mean Flows" (Geng et al., 2025)
        """
        batch_size = x.shape[0]
        import numpy as np

        # One Bernoulli decision per joint trajectory, shared across agents,
        # time, and the r/t forwards used by the consistency surrogate.
        condition_dropout_mask = None
        if self.returns_condition:
            condition_dropout_mask = (
                torch.rand((batch_size, 1, 1), device=x.device)
                >= self.condition_dropout
            )

        # Sample time pairs (t, r) using lognormal distribution for stability
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min for each pair (ensures r <= t)
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Apply flow consistency for flow_ratio fraction of samples
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=x.device)
        r = torch.tensor(r_np, device=x.device)

        # Following MADiff's approach: handle dimensions based on use_inv_dyn
        if self.use_inv_dyn:
            # For inverse dynamics mode: work with observations only (like MADiff)
            x1 = x[..., self.action_dim:]  # Target observations only
            x0 = torch.randn_like(x1)      # Noise observations only

            # MeanFlow trajectory: xt = (1-t)*x1 + t*x0
            t_expanded = t[:, None, None, None]
            r_expanded = r[:, None, None, None]
            xt = (1 - t_expanded) * x1 + t_expanded * x0

            # Apply conditioning
            xt = apply_conditioning(xt, cond)
            model_input = xt
        else:
            # For full trajectory mode: work with complete trajectories
            x1 = x
            x0 = torch.randn_like(x)

            # MeanFlow trajectory: xt = (1-t)*x1 + t*x0
            t_expanded = t[:, None, None, None]
            r_expanded = r[:, None, None, None]
            xt = (1 - t_expanded) * x1 + t_expanded * x0

            # Apply conditioning
            xt = apply_conditioning(xt, cond)
            model_input = xt

        # Conditional velocity (target for v-loss)
        v_cond = x0 - x1

        # Convert continuous t, r to discrete timesteps for the model
        discrete_t = self._discretize_time(t)
        discrete_r = self._discretize_time(r)

        if self.use_improved_meanflow:
            # ========== Improved MeanFlow (iMF) - v-loss ==========
            # Reference: "Improved Mean Flows" (Geng et al., 2025) Algorithm 1

            # Step 1: Predict instantaneous velocity v using boundary condition u(z,t,t)
            # At t=r, u(z,t,t) = v(z,t) by boundary condition
            v_pred = self.get_model_output(
                model_input,
                discrete_t,
                returns,
                env_ts,
                attention_masks,
                apply_cfg=False,
                condition_dropout_mask=condition_dropout_mask,
            )

            # Step 2 & 3: For the wrapper's simplified approach, we use a modified formulation
            # that's compatible with discrete timestep models.
            # We approximate the compound velocity V = u + (t-r) * du/dt
            # by using the prediction at discrete_r as u, and estimating the correction.

            if self.use_dispersive_loss and self.dispersive_loss_layer != "output":
                predicted, dispersive_reprs = self._forward_with_dispersive_hooks(
                    model_input,
                    discrete_r,
                    returns,
                    env_ts,
                    attention_masks,
                    apply_cfg=False,
                    condition_dropout_mask=condition_dropout_mask,
                )
            else:
                predicted = self.get_model_output(
                    model_input,
                    discrete_r,
                    returns,
                    env_ts,
                    attention_masks,
                    apply_cfg=False,
                    condition_dropout_mask=condition_dropout_mask,
                )
                dispersive_reprs = None

            # Construct compound velocity V = u + (t-r) * (v_pred - u) / (small_eps)
            # Simplified approximation: V ≈ v_pred when t ≈ r (flow consistency samples)
            # For t ≠ r samples, we interpolate towards v_pred
            time_diff = (t_expanded - r_expanded)
            # Use v_pred directly as compound velocity for v-loss
            V = predicted + time_diff * (v_pred - predicted).detach()

            # v-loss: regress V to conditional velocity
            meanflow_loss = F.mse_loss(V, v_cond)

            # Apply adaptive loss if enabled
            if self.use_adaptive_loss:
                from .meanflow.meanflow import adaptive_l2_loss
                meanflow_loss = adaptive_l2_loss(V - v_cond, gamma=self.gamma, c=self.c)
        else:
            # ========== Original MeanFlow - simplified u-loss ==========
            if self.use_dispersive_loss and self.dispersive_loss_layer != "output":
                predicted, dispersive_reprs = self._forward_with_dispersive_hooks(
                    model_input,
                    discrete_t,
                    returns,
                    env_ts,
                    attention_masks,
                    apply_cfg=False,
                    condition_dropout_mask=condition_dropout_mask,
                )
            else:
                predicted = self.get_model_output(
                    model_input,
                    discrete_t,
                    returns,
                    env_ts,
                    attention_masks,
                    apply_cfg=False,
                    condition_dropout_mask=condition_dropout_mask,
                )
                dispersive_reprs = None

            # MeanFlow loss: predict the instantaneous velocity x0 - x1
            meanflow_loss = F.mse_loss(predicted, v_cond)

            # Apply adaptive loss if enabled
            if self.use_adaptive_loss and not self.use_inv_dyn:
                from .meanflow.meanflow import adaptive_l2_loss
                meanflow_loss = adaptive_l2_loss(predicted - v_cond, gamma=self.gamma, c=self.c)

        loss_components = {'meanflow_loss': meanflow_loss}
        total_loss = meanflow_loss

        # Trajectory Dynamics Consistency (TDC) loss via frozen world model
        if self.use_wm_guidance and self.wm_guidance_module is not None and self.use_inv_dyn:
            # Reconstruct pseudo-denoised trajectory: x1_hat = xt - t * v_pred
            # model_input = xt, predicted = velocity prediction
            x1_hat = model_input - t_expanded * predicted  # [B, H, n_agents, obs_dim]

            tdc_loss, tdc_info = self.wm_guidance_module.compute_tdc_loss(
                trajectory_obs=x1_hat,
                inv_model=self.inv_model,
                share_inv=self.share_inv,
                joint_inv=self.joint_inv,
                loss_masks=loss_masks,
            )
            loss_components['tdc_loss'] = tdc_loss
            loss_components.update({f'tdc/{k}': v for k, v in tdc_info.items()})
            total_loss = total_loss + self.wm_guidance_weight * tdc_loss

        # Optional dispersive regularisation to encourage diverse velocity fields
        if self.use_dispersive_loss:
            if dispersive_reprs:
                if self.dispersive_loss_layer == "all":
                    dispersive_terms = [
                        compute_dispersive_loss(
                            flatten_representation(rep),
                            loss_type=self.dispersive_loss_type,
                            temperature=self.dispersive_loss_temperature,
                        )
                        for rep in dispersive_reprs
                    ]
                    dispersive_loss = torch.stack(dispersive_terms).mean()
                else:
                    dispersive_loss = compute_dispersive_loss(
                        flatten_representation(dispersive_reprs[0]),
                        loss_type=self.dispersive_loss_type,
                        temperature=self.dispersive_loss_temperature,
                    )
            else:
                repr_tensor = flatten_representation(predicted)
                dispersive_loss = compute_dispersive_loss(
                    repr_tensor,
                    loss_type=self.dispersive_loss_type,
                    temperature=self.dispersive_loss_temperature,
                )
            loss_components['dispersive_loss'] = dispersive_loss
            total_loss = total_loss + self.dispersive_loss_weight * dispersive_loss

        # Inverse dynamics loss (if enabled)
        if self.use_inv_dyn and not self.train_only_inv:
            inv_loss, inv_info = self.compute_inv_loss(x, loss_masks)
            loss_components.update(inv_info)
            loss_components['inv_loss'] = inv_loss
            inv_weight = getattr(self, 'inv_loss_weight', 0.5)
            total_loss = (1 - inv_weight) * meanflow_loss + inv_weight * inv_loss

        return total_loss, loss_components

    def compute_inv_loss(self, x: torch.Tensor, loss_masks: torch.Tensor,
                        legal_actions: Optional[torch.Tensor] = None):
        """Compute inverse dynamics loss compatible with GaussianDiffusion."""
        info = {}
        # Calculating inv loss
        x_t = x[:, :-1, :, self.action_dim :]
        a_t = x[:, :-1, :, : self.action_dim]
        x_t_1 = x[:, 1:, :, self.action_dim :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)
        a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)
        masks_t = loss_masks[:, 1:].reshape(-1, loss_masks.shape[2])
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(
                -1, *legal_actions.shape[2:]
            )

        if self.joint_inv or self.share_inv:
            if self.joint_inv:
                pred_a_t = self.inv_model(
                    x_comb_t.reshape(x_comb_t.shape[0], -1)  # (b a) f
                ).reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)
            else:
                pred_a_t = self.inv_model(x_comb_t)

            if legal_actions is not None:
                pred_a_t[legal_actions_t == 0] = -1e10
            if self.discrete_action:
                inv_loss = (
                    F.cross_entropy(
                        pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                        a_t.reshape(-1).long(),
                        reduction="none",
                    )
                    * masks_t.reshape(-1)
                ).mean() / masks_t.mean()
                inv_acc = (
                    (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                    .to(dtype=float)
                    .squeeze(-1)
                    * masks_t
                ).mean() / masks_t.mean()
                info["inv_acc"] = inv_acc
            else:
                inv_loss = (
                    F.mse_loss(pred_a_t, a_t, reduction="none") * masks_t.unsqueeze(-1)
                ).mean() / masks_t.mean()

        else:
            inv_loss = 0.0
            for i in range(self.n_agents):
                pred_a_t = self.inv_model[i](x_comb_t[:, i])
                if self.discrete_action:
                    inv_loss += (
                        F.cross_entropy(
                            pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                        )
                        * masks_t[:, i]
                    ).mean() / masks_t[:, i].mean()
                else:
                    inv_loss += (
                        F.mse_loss(pred_a_t, a_t[:, i]) * masks_t[:, i].unsqueeze(-1)
                    ).mean() / masks_t[:, i].mean()

        return inv_loss, info

    def conditional_sample(self, cond, returns=None, env_ts=None, horizon=None,
                          attention_masks=None, verbose=True, return_diffusion=False,
                          initial_noise=None, finetune_last_steps=0, **sample_kwargs):
        """
        Sample using MeanFlow approach with the shared model.

        Args:
            initial_noise: Optional pre-sampled noise tensor for deterministic replay.
            finetune_last_steps: If > 0, only allow gradients through the last N
                                denoising steps (DPPO-style partial fine-tuning).
                                Earlier steps run with no_grad and x is detached.
        """
        # Get batch size from conditioning
        state_key = 'state' if 'state' in cond else 'x'
        batch_size = cond.get(state_key, list(cond.values())[0]).shape[0]

        horizon = horizon or self.horizon + self.history_horizon
        device = list(cond.values())[0].device

        # Initialize noise based on use_inv_dyn mode (following MADiff logic)
        if initial_noise is not None:
            x = initial_noise
        elif self.use_inv_dyn:
            # For inverse dynamics: work with observations only
            shape = (batch_size, horizon, self.n_agents, self.observation_dim)
            x = torch.randn(shape, device=device)
        else:
            # For full trajectory: work with complete trajectories
            shape = (batch_size, horizon, self.n_agents, self.transition_dim)
            x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        # MeanFlow sampling with fewer steps
        inference_steps = sample_kwargs.get('inference_steps', self.max_denoising_steps)
        timesteps, discrete_timesteps = self._sampling_schedule(
            inference_steps, device
        )

        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()

        # Determine which step to start allowing gradients (DPPO-style)
        # finetune_last_steps=2 with 5 total steps → grad starts at step 3
        grad_start_step = len(timesteps) - finetune_last_steps if finetune_last_steps > 0 else len(timesteps)

        for i, t in enumerate(timesteps):
            # DPPO-style: detach x at the transition point from frozen→trainable
            # Use clone() instead of requires_grad_(True) to avoid leaf-variable
            # in-place operation error in apply_conditioning
            if i == grad_start_step and finetune_last_steps > 0:
                x = x.detach().clone().requires_grad_(True)
                x = x + 0  # Make x non-leaf so in-place conditioning works

            # Apply conditioning (both x and cond have matching dimensions)
            x = apply_conditioning(x, cond)

            # Expand this step's precomputed discrete model-time id to the batch.
            discrete_t = discrete_timesteps[i].expand(batch_size)

            # Predict velocity using the shared model
            if i < grad_start_step and finetune_last_steps > 0:
                # Frozen steps: no gradient
                with torch.no_grad():
                    predicted_velocity = self.get_model_output(
                        x, discrete_t, returns, env_ts, attention_masks
                    )
            else:
                # Trainable steps: allow gradient through backbone
                predicted_velocity = self.get_model_output(
                    x, discrete_t, returns, env_ts, attention_masks
                )

            # MeanFlow update step
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
            else:
                next_t = torch.tensor(0.0, device=device)

            dt = t - next_t
            x = x - dt * predicted_velocity

            progress.update({"t": t})
            if return_diffusion:
                diffusion.append(x)

        # Finally make sure conditioning is enforced
        x = apply_conditioning(x, cond)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def forward(self, cond, deterministic=False, **sample_kwargs):
        """Forward pass for sampling."""
        return self.conditional_sample(cond, **sample_kwargs)

    # -------------------------------------------------------------------------
    # Dispersive loss helpers
    # -------------------------------------------------------------------------

    def _forward_with_dispersive_hooks(
        self,
        model_input: torch.Tensor,
        discrete_t: torch.Tensor,
        returns: Optional[torch.Tensor],
        env_ts: Optional[torch.Tensor],
        attention_masks: Optional[torch.Tensor],
        apply_cfg: bool = True,
        condition_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        modules = self._select_dispersive_modules()
        representations: List[torch.Tensor] = []

        if not modules:
            predicted = self.get_model_output(
                model_input,
                discrete_t,
                returns,
                env_ts,
                attention_masks,
                apply_cfg=apply_cfg,
                condition_dropout_mask=condition_dropout_mask,
            )
            return predicted, representations

        hooks = []

        def make_hook():
            def _hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                representations.append(output)

            return _hook

        try:
            for module in modules:
                hooks.append(module.register_forward_hook(make_hook()))

            predicted = self.get_model_output(
                model_input,
                discrete_t,
                returns,
                env_ts,
                attention_masks,
                apply_cfg=apply_cfg,
                condition_dropout_mask=condition_dropout_mask,
            )
        finally:
            for hook in hooks:
                hook.remove()

        return predicted, representations

    def _select_dispersive_modules(self) -> List[nn.Module]:
        stage = getattr(self, "dispersive_loss_layer", "output")
        if stage == "output":
            return []

        model = getattr(self, "model", None)
        net = getattr(model, "net", None)
        if net is None:
            return []

        modules: List[nn.Module] = []

        def add(module: Optional[nn.Module]):
            if isinstance(module, nn.Module):
                modules.append(module)

        if stage == "early":
            downs = getattr(net, "downs", [])
            if len(downs) > 0:
                block = downs[0]
                if isinstance(block, nn.ModuleList) and len(block) > 0:
                    add(block[0])

        elif stage == "mid":
            add(getattr(net, "mid_block1", None))
            if not modules:
                downs = getattr(net, "downs", [])
                mid_idx = len(downs) // 2
                if len(downs) > mid_idx:
                    block = downs[mid_idx]
                    if isinstance(block, nn.ModuleList) and len(block) > 0:
                        add(block[0])

        elif stage == "late":
            add(getattr(net, "final_conv", None))
            if not modules:
                ups = getattr(net, "ups", [])
                if len(ups) > 0:
                    block = ups[-1]
                    if isinstance(block, nn.ModuleList) and len(block) > 0:
                        add(block[0])

        elif stage == "all":
            for collection_name in ["downs", "ups"]:
                collection = getattr(net, collection_name, [])
                for block in collection:
                    if isinstance(block, nn.ModuleList):
                        for sub_module in block:
                            add(sub_module)
            add(getattr(net, "mid_block1", None))
            add(getattr(net, "mid_block2", None))
            add(getattr(net, "final_conv", None))
            self_attn = getattr(model, "self_attn", None)
            if isinstance(self_attn, nn.ModuleList):
                for attn_module in self_attn:
                    add(attn_module)

        else:
            raise ValueError(
                "Unknown dispersive_loss_layer '{}'.".format(stage)
            )

        unique_modules: List[nn.Module] = []
        seen = set()
        for module in modules:
            identifier = id(module)
            if identifier not in seen:
                seen.add(identifier)
                unique_modules.append(module)

        return unique_modules
