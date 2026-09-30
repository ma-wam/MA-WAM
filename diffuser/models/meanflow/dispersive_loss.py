# MIT License

"""
Dispersive loss utilities for MeanFlow models.

The loss encourages sample diversity by pushing intermediate
representations away from each other. This module provides a small
standalone helper so that MeanFlow wrappers can opt-in without pulling
extra third-party dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_zero(device: torch.device) -> torch.Tensor:
    """Return a detached zero scalar on the requested device."""

    return torch.tensor(0.0, device=device)


def compute_dispersive_loss(
    representations: torch.Tensor,
    loss_type: str = "infonce_l2",
    temperature: float = 0.5,
) -> torch.Tensor:
    """Compute dispersive regularisation over a batch of features.

    Args:
        representations: Tensor shaped ``(B, F)`` containing flattened
            features for each element in the mini-batch.
        loss_type: One of ``{"infonce_l2", "infonce_cosine", "hinge", "covariance"}``.
        temperature: Temperature for InfoNCE variants.

    Returns:
        Scalar tensor representing the dispersive penalty.
    """

    batch = representations.size(0)
    if batch <= 1:
        return _safe_zero(representations.device)

    if loss_type == "infonce_l2":
        distances = torch.cdist(representations, representations, p=2)
        mask = ~torch.eye(batch, dtype=bool, device=representations.device)
        distances = distances[mask]
        exp_neg = torch.exp(-distances / temperature)
        return torch.log(torch.mean(exp_neg))

    if loss_type == "infonce_cosine":
        normed = F.normalize(representations, p=2, dim=1)
        sims = normed @ normed.t()
        distances = 1 - sims
        mask = ~torch.eye(batch, dtype=bool, device=representations.device)
        distances = distances[mask]
        exp_neg = torch.exp(-distances / temperature)
        return torch.log(torch.mean(exp_neg))

    if loss_type == "hinge":
        distances = torch.cdist(representations, representations, p=2)
        mask = ~torch.eye(batch, dtype=bool, device=representations.device)
        distances = distances[mask]
        eps = 1.0
        hinge = torch.clamp(eps - distances, min=0) ** 2
        return hinge.mean()

    if loss_type == "covariance":
        centred = representations - representations.mean(dim=0, keepdim=True)
        cov = (centred.t() @ centred) / (batch - 1)
        mask = ~torch.eye(cov.size(0), dtype=bool, device=cov.device)
        off_diag = cov[mask]
        return torch.sum(off_diag ** 2)

    raise ValueError(f"Unknown dispersive loss type: {loss_type}")


def flatten_representation(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten a trajectory or velocity tensor into ``(B, -1)`` form."""

    if tensor.dim() == 2:
        return tensor
    return tensor.view(tensor.size(0), -1)
