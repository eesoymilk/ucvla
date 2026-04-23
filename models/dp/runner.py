"""
UCVLADPRunner: training and inference wrapper for the UCVLADiT proxy model.

Loss design:
  L = L_flow_matching
    + lambda_kl    · L_KL          (variational embedding regularization)
    + lambda_triplet · L_triplet   (push user z-vectors apart in embedding space)
    + lambda_sdtw  · L_sdtw        (windowed soft-DTW triplet on predicted velocities;
                                    position-agnostic — finds discriminative segment
                                    regardless of where in the trajectory users differ)
    + lambda_ortho · L_ortho       (decorrelate user mu vectors)

All terms except L_flow_matching are controlled by their lambda — set to 0.0
to disable for ablation.
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
        Only UCVLADiT.user_bias (mu + log_var) and UCVLADiT.bias_proj.
        Call freeze_base() to lock everything else.

    Args:
        model:               UCVLADiT instance.
        num_inference_steps: ODE integration steps at inference.
        lambda_triplet:      Weight for embedding-space triplet loss.
        lambda_ortho:        Weight for orthogonality loss on mu vectors.
        lambda_kl:           Weight for KL divergence (variational regularization).
        lambda_sdtw:         Weight for windowed soft-DTW triplet on pred velocities.
        triplet_margin:      Margin for both triplet losses.
        sdtw_window:         Window size for soft-DTW triplet (timesteps).
        sdtw_stride:         Stride for sliding window.
        sdtw_gamma:          Smoothing parameter for soft-DTW (lower = sharper).
    """

    def __init__(
        self,
        model: UCVLADiT,
        num_inference_steps: int = 10,
        lambda_triplet: float = 0.1,
        lambda_ortho: float = 0.01,
        lambda_kl: float = 0.001,
        lambda_sdtw: float = 0.0,
        triplet_margin: float = 0.5,
        sdtw_window: int = 8,
        sdtw_stride: int = 4,
        sdtw_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.lambda_triplet = lambda_triplet
        self.lambda_ortho = lambda_ortho
        self.lambda_kl = lambda_kl
        self.lambda_sdtw = lambda_sdtw
        self.triplet_margin = triplet_margin
        self.sdtw_window = sdtw_window
        self.sdtw_stride = sdtw_stride
        self.sdtw_gamma = sdtw_gamma

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
    # Auxiliary losses                                                     #
    # ------------------------------------------------------------------ #

    def _kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL( N(mu, sigma²) || N(0,I) ) averaged over batch and dims."""
        return (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean()

    def _triplet_loss(self, bias: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
        """
        Push same-user z-vectors together, cross-user apart (embedding space).

        For each sample i, picks a random positive (same user_id) and a random
        negative (different user_id). Falls back gracefully when a user appears
        only once in the batch.
        """
        B = bias.shape[0]
        device = bias.device
        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(B):
            uid = user_id[i].item()
            pos_mask = (user_id == uid).clone()
            pos_mask[i] = False
            neg_mask = user_id != uid

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
        """Encourage all user mu vectors to be mutually orthogonal."""
        B_mat = self.model.user_bias.weight          # (n_users, d_bias) — mu
        B_norm = F.normalize(B_mat, dim=-1)
        gram = B_norm @ B_norm.T
        eye = torch.eye(gram.shape[0], device=gram.device)
        return (gram - eye).norm()

    def _soft_dtw_batch(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Batched soft-DTW distance (pure PyTorch, no extra deps).

        Args:
            a: (B, T_a, D)
            b: (B, T_b, D)

        Returns:
            (B,) soft-DTW distances.
        """
        B, T_a, _ = a.shape
        T_b = b.shape[1]
        gamma = self.sdtw_gamma

        dist = torch.cdist(a, b).pow(2)  # (B, T_a, T_b)

        R = torch.full(
            (B, T_a + 1, T_b + 1), float("inf"), device=a.device, dtype=a.dtype
        )
        R[:, 0, 0] = 0.0

        for i in range(1, T_a + 1):
            for j in range(1, T_b + 1):
                neighbors = torch.stack(
                    [R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]], dim=1
                )  # (B, 3)
                soft_min = -gamma * torch.logsumexp(-neighbors / gamma, dim=1)
                R[:, i, j] = dist[:, i - 1, j - 1] + soft_min

        return R[:, T_a, T_b]  # (B,)

    def _sdtw_triplet_loss(
        self, pred: torch.Tensor, user_id: torch.Tensor
    ) -> torch.Tensor:
        """Windowed soft-DTW triplet on predicted velocities.

        Slides a window over the T-step trajectory and takes the most
        discriminative window for the triplet, so the loss concentrates on
        wherever in the sequence users differ — no position assumption.

        Args:
            pred:    (B, T, action_dim) model velocity predictions.
            user_id: (B,) user indices.
        """
        B, T, _ = pred.shape
        device = pred.device
        w = self.sdtw_window
        starts = list(range(0, T - w + 1, self.sdtw_stride))

        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(B):
            uid = user_id[i].item()
            pos_mask = (user_id == uid).clone()
            pos_mask[i] = False
            neg_mask = user_id != uid

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_idx = pos_mask.nonzero(as_tuple=True)[0]
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            j = pos_idx[torch.randint(len(pos_idx), (1,), device=device)].item()
            k = neg_idx[torch.randint(len(neg_idx), (1,), device=device)].item()

            d_pos_per_window = []
            d_neg_per_window = []
            for s in starts:
                a_win = pred[i, s : s + w].unsqueeze(0)  # (1, w, D)
                p_win = pred[j, s : s + w].unsqueeze(0)
                n_win = pred[k, s : s + w].unsqueeze(0)
                d_pos_per_window.append(self._soft_dtw_batch(a_win, p_win))  # (1,)
                d_neg_per_window.append(self._soft_dtw_batch(a_win, n_win))

            d_pos = torch.stack(d_pos_per_window).min()
            d_neg = torch.stack(d_neg_per_window).max()
            loss = loss + F.relu(d_pos - d_neg + self.triplet_margin)
            count += 1

        return loss / max(count, 1)

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

        # Flow matching
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

        if user_id is not None:
            # KL regularization on variational embedding
            if self.lambda_kl > 0.0:
                mu, log_var = self.model.user_bias.get_distribution(user_id)
                kl = self._kl_loss(mu, log_var)
                total = total + self.lambda_kl * kl
                log["loss/kl"] = kl.item()

            # Embedding-space triplet (on sampled z)
            if self.lambda_triplet > 0.0:
                z = self.model.user_bias(user_id)
                trip = self._triplet_loss(z, user_id)
                total = total + self.lambda_triplet * trip
                log["loss/triplet"] = trip.item()

            # Windowed soft-DTW triplet on predicted velocities
            if self.lambda_sdtw > 0.0:
                sdtw = self._sdtw_triplet_loss(pred, user_id)
                total = total + self.lambda_sdtw * sdtw
                log["loss/sdtw"] = sdtw.item()

            # Orthogonality on mu vectors
            if self.lambda_ortho > 0.0:
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
