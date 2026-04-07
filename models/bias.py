"""
Per-user bias modules for UCVLA.

Ported from models/ucvla/ucvla_rdt.py in RDT2 — made standalone so they can
be used with any backbone (DP, RDT-2, π₀).

UserBias  : nn.Embedding table, zero-initialized so the model starts at base
            behavior.  Only user_bias + bias_proj are trainable in Stage 1.
BiasProj  : single nn.Linear projecting the bias vector into the backbone's
            conditioning dimension (hidden_size, d_time, etc.).
"""

import torch
import torch.nn as nn


class UserBias(nn.Module):
    """Per-user embedding table.

    Args:
        n_users:  Number of distinct users.
        d_bias:   Dimension of each user's bias vector.
    """

    def __init__(self, n_users: int, d_bias: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_users, d_bias)
        nn.init.zeros_(self.embedding.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, user_id: torch.Tensor) -> torch.Tensor:
        """Return bias vectors for the given user IDs.

        Args:
            user_id: (B,) LongTensor of user indices.

        Returns:
            (B, d_bias) bias vectors.
        """
        return self.embedding(user_id)


class BiasProj(nn.Module):
    """Project user bias into the backbone conditioning dimension.

    Args:
        d_bias:       Input dimension (must match UserBias.d_bias).
        hidden_size:  Output dimension (hidden_size / d_time of the backbone).
    """

    def __init__(self, d_bias: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_bias, hidden_size)
        nn.init.zeros_(self.linear.bias)

    def forward(self, bias: torch.Tensor) -> torch.Tensor:
        """Project bias vectors.

        Args:
            bias: (B, d_bias) output of UserBias.

        Returns:
            (B, hidden_size) projected bias.
        """
        return self.linear(bias)
