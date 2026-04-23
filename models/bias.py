"""
Per-user bias modules for UCVLA.

Ported from models/ucvla/ucvla_rdt.py in RDT2 — made standalone so they can
be used with any backbone (DP, RDT-2, π₀).

VariationalUserBias : variational embedding (mu + log_var) so the user bias is
                      a distribution from Stage 1.  Stage 2's bias predictor
                      then learns to output the same (mu, sigma) space from
                      passive observations — no backbone retraining needed.
BiasProj            : single nn.Linear projecting the bias vector into the
                      backbone's conditioning dimension (hidden_size, d_time).
"""

import torch
import torch.nn as nn


class VariationalUserBias(nn.Module):
    """Per-user variational embedding (mean + log-variance).

    Makes the user embedding a distribution from Stage 1 so Stage 2's bias
    predictor can target the same (mu, sigma) space without retraining.

    During training:  z = mu + eps * sigma  (reparameterization trick)
    During eval:      z = mu                (deterministic mean)

    Args:
        n_users: Number of distinct users.
        d_bias:  Dimension of each user's bias vector.
    """

    def __init__(self, n_users: int, d_bias: int = 64) -> None:
        super().__init__()
        self.mu_embed = nn.Embedding(n_users, d_bias)
        self.log_var_embed = nn.Embedding(n_users, d_bias)
        nn.init.normal_(self.mu_embed.weight, std=0.02)
        nn.init.constant_(self.log_var_embed.weight, -2.0)  # sigma ≈ 0.37 initially

    @property
    def weight(self) -> torch.Tensor:
        """Return mu weights — used by ortho loss and _init_weights."""
        return self.mu_embed.weight

    def get_distribution(
        self, user_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) for the given user IDs.

        Args:
            user_id: (B,) LongTensor of user indices.

        Returns:
            mu:      (B, d_bias)
            log_var: (B, d_bias)
        """
        return self.mu_embed(user_id), self.log_var_embed(user_id)

    def forward(self, user_id: torch.Tensor) -> torch.Tensor:
        """Sample bias vector (training) or return mu (eval).

        Args:
            user_id: (B,) LongTensor of user indices.

        Returns:
            (B, d_bias) bias vectors.
        """
        mu, log_var = self.get_distribution(user_id)
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + torch.randn_like(std) * std
        return mu


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
