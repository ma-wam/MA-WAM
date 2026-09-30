import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_dynamics import ExpertMLP, SoftMoEDynamics
from .moe_reward import SparseMoEReward


class SparseMoEDynamics(nn.Module):
    """Token-wise sparse expert dynamics used for routing ablations."""

    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        n_experts=8,
        hidden_dim=256,
        top_k=2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(self.input_dim, n_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(self.input_dim, hidden_dim, obs_dim) for _ in range(n_experts)]
        )
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.output_norm = nn.LayerNorm(obs_dim)

    def _route(self, tokens):
        batch_size = tokens.shape[0]
        flat = tokens.reshape(batch_size * self.n_agents, self.input_dim)
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        all_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        selected = torch.gather(
            all_outputs,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.obs_dim),
        )
        output = (top_k_probs.unsqueeze(-1) * selected).sum(dim=1)
        output = output.reshape(batch_size, self.n_agents, self.obs_dim)

        expert_mask = F.one_hot(top_k_indices, self.n_experts).float()
        expert_usage = expert_mask.sum(dim=1).mean(dim=0)
        expert_prob_mean = router_probs.mean(dim=0)
        aux_loss = (expert_usage * expert_prob_mean).sum() * self.n_experts
        return output, aux_loss

    def forward(self, obs, act):
        next_obs, _ = self.forward_with_aux(obs, act)
        return next_obs

    def forward_with_aux(self, obs, act):
        tokens = torch.cat([obs, act], dim=-1)
        tokens = self.input_norm(tokens)
        output, aux_loss = self._route(tokens)
        output = self.output_norm(output)
        return obs + output, aux_loss

    def forward_with_routing(self, obs, act):
        tokens = torch.cat([obs, act], dim=-1)
        tokens = self.input_norm(tokens)
        batch_size = tokens.shape[0]
        flat = tokens.reshape(batch_size * self.n_agents, self.input_dim)
        router_probs = F.softmax(self.router(flat), dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        all_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        selected = torch.gather(
            all_outputs,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.obs_dim),
        )
        output = (top_k_probs.unsqueeze(-1) * selected).sum(dim=1)
        output = self.output_norm(output.reshape(batch_size, self.n_agents, self.obs_dim))
        return obs + output, {
            "router_probs": router_probs,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs,
        }


class SoftMoEReward(nn.Module):
    """Soft-routed reward model used for the all-soft routing ablation."""

    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        n_experts=4,
        hidden_dim=256,
        n_slots=4,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim * 2 + act_dim
        self.n_experts = n_experts
        self.n_slots = n_slots
        total_slots = n_experts * n_slots

        self.dispatch_weights = nn.Parameter(torch.randn(self.input_dim, total_slots) * 0.02)
        self.combine_weights = nn.Parameter(torch.randn(self.input_dim, total_slots) * 0.02)
        self.experts = nn.ModuleList(
            [ExpertMLP(self.input_dim, hidden_dim, 1) for _ in range(n_experts)]
        )
        self.input_norm = nn.LayerNorm(self.input_dim)

    def forward(self, obs, act, next_obs):
        tokens = torch.cat([obs, act, next_obs], dim=-1)
        tokens = self.input_norm(tokens)

        dispatch_logits = torch.matmul(tokens, self.dispatch_weights)
        dispatch_probs = F.softmax(dispatch_logits, dim=1)
        slot_inputs = torch.bmm(dispatch_probs.transpose(1, 2), tokens)

        slot_outputs = []
        for i, expert in enumerate(self.experts):
            start = i * self.n_slots
            end = start + self.n_slots
            expert_input = slot_inputs[:, start:end]
            b, s, d = expert_input.shape
            slot_outputs.append(expert(expert_input.reshape(b * s, d)).reshape(b, s, 1))
        all_slot_outputs = torch.cat(slot_outputs, dim=1)

        combine_logits = torch.matmul(tokens, self.combine_weights)
        combine_probs = F.softmax(combine_logits, dim=-1)
        reward_pred = torch.bmm(combine_probs, all_slot_outputs)
        aux_loss = reward_pred.new_zeros(())
        return reward_pred, aux_loss

    def forward_with_routing(self, obs, act, next_obs):
        tokens = torch.cat([obs, act, next_obs], dim=-1)
        tokens = self.input_norm(tokens)
        dispatch_logits = torch.matmul(tokens, self.dispatch_weights)
        dispatch_probs = F.softmax(dispatch_logits, dim=1)
        slot_inputs = torch.bmm(dispatch_probs.transpose(1, 2), tokens)

        per_expert_outputs = []
        slot_outputs = []
        for i, expert in enumerate(self.experts):
            start = i * self.n_slots
            end = start + self.n_slots
            expert_input = slot_inputs[:, start:end]
            b, s, d = expert_input.shape
            expert_output = expert(expert_input.reshape(b * s, d)).reshape(b, s, 1)
            per_expert_outputs.append(expert_output)
            slot_outputs.append(expert_output)
        all_slot_outputs = torch.cat(slot_outputs, dim=1)

        combine_logits = torch.matmul(tokens, self.combine_weights)
        combine_probs = F.softmax(combine_logits, dim=-1)
        reward_pred = torch.bmm(combine_probs, all_slot_outputs)
        return reward_pred, reward_pred.new_zeros(()), {
            "dispatch_probs": dispatch_probs,
            "combine_probs": combine_probs,
            "per_expert_outputs": per_expert_outputs,
        }


class RoutingVariantWorldModel(nn.Module):
    """World-model variants for HCR-WM routing ablations."""

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
        variant="soft_sparse",
    ):
        super().__init__()
        if variant not in {"soft_sparse", "soft_only", "sparse_only", "sparse_soft"}:
            raise ValueError(f"unknown routing variant: {variant}")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots
        self.variant = variant
        self.aux_loss_weight = aux_loss_weight

        if variant in {"soft_sparse", "soft_only"}:
            self.dynamics = SoftMoEDynamics(
                n_agents=n_agents,
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_experts=n_dyn_experts,
                hidden_dim=hidden_dim,
                n_slots=n_slots,
            )
        else:
            self.dynamics = SparseMoEDynamics(
                n_agents=n_agents,
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_experts=n_dyn_experts,
                hidden_dim=hidden_dim,
                top_k=2,
            )

        if variant in {"soft_only", "sparse_soft"}:
            self.reward_predictor = SoftMoEReward(
                n_agents=n_agents,
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_experts=n_rew_experts,
                hidden_dim=hidden_dim,
                n_slots=n_slots,
            )
        else:
            self.reward_predictor = SparseMoEReward(
                n_agents=n_agents,
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_experts=n_rew_experts,
                hidden_dim=hidden_dim,
            )

    def _dynamics_forward_train(self, obs, act):
        if hasattr(self.dynamics, "forward_with_aux"):
            return self.dynamics.forward_with_aux(obs, act)
        next_obs = self.dynamics(obs, act)
        return next_obs, next_obs.new_zeros(())

    def forward(self, obs, act):
        next_obs_pred = self.dynamics(obs, act)
        reward_pred, _ = self.reward_predictor(obs, act, next_obs_pred.detach())
        return next_obs_pred, reward_pred

    def loss(self, obs, act, next_obs_target, reward_target):
        if reward_target.dim() == 2:
            reward_target = reward_target.unsqueeze(-1)
        next_obs_pred, dyn_aux = self._dynamics_forward_train(obs, act)
        dyn_loss = F.mse_loss(next_obs_pred, next_obs_target)
        reward_pred, rew_aux = self.reward_predictor(obs, act, next_obs_pred.detach())
        rew_loss = F.mse_loss(reward_pred, reward_target)
        aux_loss = dyn_aux + rew_aux
        total_loss = dyn_loss + rew_loss + self.aux_loss_weight * aux_loss
        return total_loss, {
            "dynamics_loss": dyn_loss.item(),
            "reward_loss": rew_loss.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item(),
        }

    def predict_next_obs(self, obs, act):
        return self.dynamics(obs, act)

    def predict_reward(self, obs, act, next_obs):
        reward_pred, _ = self.reward_predictor(obs, act, next_obs)
        return reward_pred
