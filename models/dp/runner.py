"""
UCVLADPRunner: training and inference wrapper for the UCVLADiT proxy model.

Loss design (from PROXY_PLAN.md §6):
  1. MSE only (baseline)
  2. MSE + triplet
  3. MSE + ortho
  4. MSE + triplet + ortho

Ablation is controlled by lambda_triplet / lambda_ortho in the config —
set either to 0.0 to disable.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dp.model import UCVLADiT


class UCVLADPRunner(nn.Module):
    """
    Training/inference wrapper around UCVLADiT.

    Trainable parameters (Stage 1):
        Only UCVLADiT.user_bias and UCVLADiT.bias_proj.
        Call freeze_base() to lock everything else.

    Flow-matching convention:
        noisy_action = action_gt * t + noise * (1 - t)
        target       = action_gt - noise  (velocity)
        loss         = MSE(model_output, target)

    Args:
        model:            UCVLADiT instance.
        num_inference_steps: ODE integration steps at inference.
        lambda_triplet:   Weight for triplet loss (0.0 = disabled).
        lambda_ortho:     Weight for orthogonality loss (0.0 = disabled).
        triplet_margin:   Margin for triplet loss.
    """

    def __init__(
        self,
        model: UCVLADiT,
        num_inference_steps: int = 10,
        lambda_triplet: float = 0.1,
        lambda_ortho: float = 0.01,
        triplet_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.lambda_triplet = lambda_triplet
        self.lambda_ortho = lambda_ortho
        self.triplet_margin = triplet_margin

    # ------------------------------------------------------------------ #
    # Parameter management                                                 #
    # ------------------------------------------------------------------ #

    def freeze_base(self) -> None:
        """Freeze all backbone weights; only user_bias + bias_proj remain trainable."""
        for name, p in self.model.named_parameters():
            if name.startswith("user_bias") or name.startswith("bias_proj"):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.model.user_bias.parameters())
            + list(self.model.bias_proj.parameters())
        )

    # ------------------------------------------------------------------ #
    # UCVLA auxiliary losses                                               #
    # ------------------------------------------------------------------ #

    def _triplet_loss(self, bias: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
        """
        Push same-user bias vectors together, cross-user apart.

        For each sample i, we pick a random positive (another sample with the
        same user_id) and a random negative (a sample with a different user_id).
        Falls back gracefully when a user appears only once in the batch.
        """
        B = bias.shape[0]
        device = bias.device
        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(B):
            uid = user_id[i].item()
            pos_mask = (user_id == uid)
            pos_mask[i] = False
            neg_mask = (user_id != uid)

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_idx = pos_mask.nonzero(as_tuple=True)[0]
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            j = pos_idx[torch.randint(len(pos_idx), (1,), device=device)].item()
            k = neg_idx[torch.randint(len(neg_idx), (1,), device=device)].item()

            loss = loss + F.triplet_margin_loss(
                bias[i].unsqueeze(0),
                bias[j].unsqueeze(0),
                bias[k].unsqueeze(0),
                margin=self.triplet_margin,
            )
            count += 1

        return loss / max(count, 1)

    def _ortho_loss(self) -> torch.Tensor:
        """Encourage all user bias vectors to be mutually orthogonal."""
        B_mat = self.model.user_bias.weight          # (n_users, bias_dim)
        B_norm = F.normalize(B_mat, dim=-1)
        gram = B_norm @ B_norm.T                      # (n_users, n_users)
        eye = torch.eye(gram.shape[0], device=gram.device)
        return (gram - eye).norm()

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        action_gt: torch.Tensor,
        clip_tokens: torch.Tensor,
        state: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined UCVLA loss.

        Args:
            action_gt:   (B, T, action_dim) ground-truth actions.
            clip_tokens: (B, 196, 512) frozen CLIP patch tokens.
            state:       (B, 1, state_dim) proprioceptive state.
            user_id:     (B,) LongTensor of user indices.

        Returns:
            (total_loss, log_dict) where log_dict has per-component values.
        """
        B, T, A = action_gt.shape
        device = action_gt.device
        dtype = action_gt.dtype

        # Flow matching noise / timestep
        noise = torch.randn_like(action_gt)
        t = torch.rand(B, device=device, dtype=dtype)
        noisy_action = action_gt * t.view(B, 1, 1) + noise * (1 - t.view(B, 1, 1))
        target = action_gt - noise

        pred = self.model(
            noisy_action=noisy_action,
            t=t,
            clip_tokens=clip_tokens,
            state=state,
            user_id=user_id,
        )
        mse = F.mse_loss(pred, target)
        total = mse
        log: dict[str, float] = {"loss/mse": mse.item()}

        if self.lambda_triplet > 0.0 and user_id is not None:
            bias = self.model.user_bias(user_id)         # (B, bias_dim)
            trip = self._triplet_loss(bias, user_id)
            total = total + self.lambda_triplet * trip
            log["loss/triplet"] = trip.item()

        if self.lambda_ortho > 0.0 and user_id is not None:
            ortho = self._ortho_loss()
            total = total + self.lambda_ortho * ortho
            log["loss/ortho"] = ortho.item()

        log["loss/total"] = total.item()
        return total, log

    def forward(self, *args, **kwargs):
        loss, _ = self.compute_loss(*args, **kwargs)
        return loss

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        clip_tokens: torch.Tensor,
        state: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run Euler ODE to predict personalized action trajectory.

        Args:
            clip_tokens: (B, 196, 512) CLIP patch tokens.
            state:       (B, 1, state_dim) proprioceptive state.
            user_id:     (B,) LongTensor of user indices, or None.

        Returns:
            (B, T, action_dim) predicted action trajectory.
        """
        B = clip_tokens.shape[0]
        device = clip_tokens.device
        dtype = clip_tokens.dtype

        x = torch.randn(
            B, self.model.pred_horizon, self.model.action_dim,
            device=device, dtype=dtype,
        )
        step_size = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * step_size, device=device, dtype=dtype)
            vel = self.model(
                noisy_action=x,
                t=t,
                clip_tokens=clip_tokens,
                state=state,
                user_id=user_id,
            )
            x = x + vel * step_size

        return x
