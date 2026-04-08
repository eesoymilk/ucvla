"""
Minimal DiT (Diffusion Transformer) diffusion policy for UCVLA proxy.

Architecture
------------
image  → frozen CLIP-ViT-B/16 → [B, 196, 512] → clip_proj  → [B, 196, D]  ┐
state  → state_embed           → [B, 1, D]                                   ├─ context
                                                                              ┘
noisy_action → action_embed → [B, T, D]
t_embed(t) + bias_proj(user_bias[user_id]) → cond [B, D] → adaLN each block

Each DiTBlock:
  1. adaLN self-attention (action tokens attend to each other)
  2. adaLN cross-attention (action tokens attend to context)
  3. adaLN FFN

Output head: LayerNorm → Linear → [B, T, action_dim]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: (1 + scale) * LayerNorm(x) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(freq)


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.num_heads).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h t d -> b t (h d)")
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (h d) -> b h t d", h=self.num_heads)
        kv = self.kv(context)
        k, v = rearrange(kv, "b s (two h d) -> two b h s d", two=2, h=self.num_heads).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h t d -> b t (h d)")
        return self.proj(x)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    DiT block with adaLN conditioning, self-attention, cross-attention, and FFN.

    Conditioning vector ``cond`` (B, D) provides 9 adaLN parameters:
    shift/scale for (self-attn norm, cross-attn norm, FFN norm) = 6,
    plus gate for each = 3 gates → 9 total via a single linear.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.self_attn = SelfAttention(hidden_size, num_heads)
        self.cross_attn = CrossAttention(hidden_size, num_heads)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN: 9 params (shift_sa, scale_sa, gate_sa,
        #                   shift_ca, scale_ca, gate_ca,
        #                   shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) action token sequence.
            cond:    (B, D) conditioning vector (t_embed + bias).
            context: (B, S, D) cross-attention context (CLIP patches + state).

        Returns:
            (B, T, D) updated action tokens.
        """
        mods = self.adaLN_modulation(cond).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff = mods

        # Self-attention
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        # Cross-attention
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), context)
        # FFN
        x = x + gate_ff.unsqueeze(1) * self.ffn(modulate(self.norm3(x), shift_ff, scale_ff))
        return x


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """adaLN + linear output projection."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------

class UCVLADiT(nn.Module):
    """
    Lightweight DiT diffusion policy conditioned on per-user bias.

    The base backbone (everything except user_bias + bias_proj) is ~10M params.
    Only user_bias and bias_proj are trained in Stage 1.

    Args:
        n_users:      Number of distinct users.
        bias_dim:     Per-user bias vector dimension.
        hidden_size:  DiT hidden dimension.
        depth:        Number of DiT blocks.
        num_heads:    Number of attention heads.
        pred_horizon: Action prediction horizon (T).
        action_dim:   Action dimensionality.
        state_dim:    State dimensionality.
        clip_dim:     CLIP patch token dimension (512 for ViT-B/16).
    """

    def __init__(
        self,
        n_users: int,
        bias_dim: int = 64,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        pred_horizon: int = 24,
        action_dim: int = 20,
        state_dim: int = 14,
        clip_dim: int = 768,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # --- Trainable UCVLA bias parameters (Stage 1 only) ---
        self.user_bias = nn.Embedding(n_users, bias_dim)
        self.bias_proj = nn.Linear(bias_dim, hidden_size)
        nn.init.normal_(self.user_bias.weight, std=0.02)
        nn.init.zeros_(self.bias_proj.bias)

        # --- Timestep embedding ---
        self.t_embedder = TimestepEmbedder(hidden_size)

        # --- Input projections ---
        self.action_embed = nn.Linear(action_dim, hidden_size)
        self.clip_proj = nn.Linear(clip_dim, hidden_size)
        self.state_embed = nn.Linear(state_dim, hidden_size)

        # --- DiT backbone ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Zero-init adaLN modulation layers so identity at start
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # Zero-init output projection
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # Re-init bias params after apply (they have their own init)
        nn.init.normal_(self.user_bias.weight, std=0.02)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        clip_tokens: torch.Tensor,
        state: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict denoised action velocity.

        Args:
            noisy_action: (B, T, action_dim) noisy action trajectory.
            t:            (B,) or (1,) diffusion timesteps in [0, 1].
            clip_tokens:  (B, 196, 512) frozen CLIP patch tokens.
            state:        (B, 1, state_dim) proprioceptive state (can be zeros).
            user_id:      (B,) LongTensor of user indices, or None.

        Returns:
            (B, T, action_dim) predicted velocity (flow matching target).
        """
        B = noisy_action.shape[0]
        if t.shape[0] == 1:
            t = t.expand(B)

        # Conditioning: t_embed + optional user bias
        cond = self.t_embedder(t)  # (B, D)
        if user_id is not None:
            cond = cond + self.bias_proj(self.user_bias(user_id))  # (B, D)

        # Context tokens: CLIP patches + state
        ctx_clip = self.clip_proj(clip_tokens)           # (B, 196, D)
        ctx_state = self.state_embed(state)              # (B, 1, D)
        context = torch.cat([ctx_clip, ctx_state], dim=1)  # (B, 197, D)

        # Action token sequence
        x = self.action_embed(noisy_action)  # (B, T, D)

        for block in self.blocks:
            x = block(x, cond, context)

        return self.final_layer(x, cond)  # (B, T, action_dim)
