# =====================================================================
# Audio Spectrogram Transformer (AST) Model
# Modified to return CLS token embedding alongside classification logits
# so that predictions.csv can include feature vectors for the Fusion Layer.
# Supports gradient checkpointing for memory-efficient training.
# =====================================================================

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


class AudioSpectrogramTransformer(nn.Module):
    """
    A lightweight Vision-Transformer-style model that operates on 2D
    Mel-spectrogram patches. Returns both classification logits and the
    CLS token embedding used for prediction.
    """

    def __init__(
        self,
        n_mels: int = 128,
        max_frames: int = 512,
        patch_size: tuple = (16, 16),
        num_classes: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # -----------------------------------------------------------------
        # Compute the 2D patch grid dimensions
        # -----------------------------------------------------------------
        f_dim = n_mels // patch_size[0]       # 8  when n_mels=128, patch=16
        t_dim = max_frames // patch_size[1]   # 12 when max_frames=192, patch=16
        self.num_patches = f_dim * t_dim      # 96 patches at 128×192

        # -----------------------------------------------------------------
        # Patch projection via non-overlapping 2D convolution
        # -----------------------------------------------------------------
        self.patch_proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # -----------------------------------------------------------------
        # Learnable tokens
        # -----------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )

        # -----------------------------------------------------------------
        # Transformer encoder (pre-norm style)
        # -----------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # -----------------------------------------------------------------
        # Classification head (operates on CLS token only)
        # -----------------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # -----------------------------------------------------------------
    # Freeze a specific encoder layer (for ablation experiments)
    # -----------------------------------------------------------------
    def freeze_encoder_layer(self, layer_idx: int):
        """Freeze one encoder layer. 0 = first, -1 = last."""
        layer = self.encoder.layers[layer_idx]
        frozen = 0
        for param in layer.parameters():
            param.requires_grad = False
            frozen += param.numel()
        print(f"  Frozen encoder layer {layer_idx} ({frozen:,} params)")

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N_MELS, MAX_FRAMES] — batch of Mel spectrograms

        Returns:
            logits:       [B, num_classes] — classification scores
            cls_embedding: [B, d_model]    — raw CLS token before the head
        """
        # Add channel dimension → [B, 1, F, T]
        x = x.unsqueeze(1)

        # Patch projection → [B, num_patches, d_model]
        h = self.patch_proj(x).flatten(2).transpose(1, 2)

        # Prepend CLS token
        B = h.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, num_patches+1, d_model]

        # Add positional encoding
        h = h + self.pos_embed

        # Transformer encoder (with optional gradient checkpointing per layer)
        if self.use_checkpoint and self.training:
            for layer in self.encoder.layers:
                h = _grad_checkpoint(layer, h, use_reentrant=False)
        else:
            h = self.encoder(h)

        # Extract CLS embedding and classify
        cls_embedding = self.norm(h[:, 0])        # [B, d_model]
        logits = self.cls_head(cls_embedding)     # [B, num_classes]

        return logits, cls_embedding
