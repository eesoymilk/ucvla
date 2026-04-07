"""
Frozen CLIP ViT-B/16 vision encoder.

Returns spatial patch tokens (196 patches, 768-D each) for use as
cross-attention context in the DiT diffusion policy.

The encoder is fully frozen — no gradients flow through it during training.

Note on dimensions:
  open_clip ViT-B/16 internal hidden dim = 768.
  The final 512-D embedding is produced by projecting the CLS token.
  We return raw patch tokens (768-D) before that projection.
"""

import torch
import torch.nn as nn

try:
    import open_clip
except ImportError as e:
    raise ImportError("open-clip-torch is required: uv add open-clip-torch") from e


class CLIPEncoder(nn.Module):
    """Frozen CLIP ViT-B/16 returning patch tokens.

    Input:  (B, 3, 224, 224) images, preprocessed with ``get_transform()``.
    Output: (B, 196, 768) patch tokens (CLS token dropped).

    The model and preprocessor (``transform``) are loaded on construction.
    Call ``get_transform()`` to obtain the torchvision-compatible transform
    that should be applied before calling ``forward()``.
    """

    PATCH_DIM: int = 768    # ViT-B/16 transformer hidden dimension
    N_PATCHES: int = 196    # 14×14 patches from 224×224 input

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai") -> None:
        super().__init__()
        model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        # Freeze everything
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.eval()

    # Stay frozen even if parent calls .train()
    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    def get_transform(self):
        """Return the CLIP preprocessing transform (torchvision-compatible)."""
        return self._preprocess

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens.

        Args:
            x: (B, 3, 224, 224) preprocessed images.

        Returns:
            (B, 196, 768) spatial patch tokens.
        """
        vt = self.visual

        # Patch embedding: conv1 → (B, D, 14, 14) → (B, 196, D)
        x = vt.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # Prepend CLS token
        cls = vt.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Positional embedding
        x = x + vt.positional_embedding

        # Pre-norm (if present)
        x = vt.ln_pre(x)

        # Transformer blocks
        x = vt.transformer(x)

        # Drop CLS token, return patch tokens
        return x[:, 1:, :]  # (B, 196, 768)
