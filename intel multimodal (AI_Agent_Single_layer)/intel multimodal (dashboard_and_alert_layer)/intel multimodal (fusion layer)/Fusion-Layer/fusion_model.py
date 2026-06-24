"""
fusion_model.py — Multimodal Transformer Encoder with cross-modal attention.

Architecture:
  1.  Input: [B, 1024]  concatenated features from Vision(768) + Audio(128) + Physio(128).
  2.  Tokenization: each modality projected to d_model via separate Linear layers.
  3.  [CLS] token prepended → [B, 4, d_model].
  4.  Learnable positional embedding added.
  5.  Transformer Encoder (n_layers × MHA + FFN) — self-attention across all 4 tokens
      inherently models cross-modal interactions (Vision↔Audio, Vision↔Physio, etc.).
  6.  CLS token → classifier head → logits.
  7.  Returns (logits, fusion_embedding) matching all three monomodal layers.

Supports gradient checkpointing and layer freezing for memory-efficient training.

Note: A forecasting head is architected but not enabled in this competition,
because the dataset lacks true longitudinal sequences.  The forecast_horizon
parameter in __init__ and the optional forecast_head attribute are kept as
placeholders for future extension when longitudinal multimodal data is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


class MultimodalFusionEncoder(nn.Module):
    """Multimodal Transformer Encoder for health state classification.

    Args:
        vision_dim:      Vision feature dimension (768 from Swin-T).
        audio_dim:       Audio feature dimension (128 from AST).
        physio_dim:      Physiological feature dimension (128 from iTransformer).
        d_model:         Internal model dimension for all tokens (default 256).
        n_heads:         Number of attention heads (default 8).
        n_layers:        Number of Transformer encoder layers (default 4).
        d_ff:            Feed-forward hidden dimension (default 512).
        dropout:         Dropout rate (default 0.1).
        num_classes:     Number of output classes (3 or 2).
        use_checkpoint:  If True, apply gradient checkpointing per encoder layer.
        forecast_horizon: Placeholder for future forecasting head (0 = disabled).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        audio_dim: int = 128,
        physio_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
        use_checkpoint: bool = False,
        forecast_horizon: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.forecast_horizon = forecast_horizon

        # -----------------------------------------------------------------
        # Modality token projections (each modality → d_model)
        # -----------------------------------------------------------------
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.physio_proj = nn.Linear(physio_dim, d_model)

        # -----------------------------------------------------------------
        # Learnable CLS token + positional embedding (4 positions)
        # -----------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.num_tokens = 4  # CLS + Vision + Audio + Physio
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, d_model) * 0.02
        )

        # -----------------------------------------------------------------
        # Transformer Encoder (pre-norm style with GELU)
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
        # Classification head (operates on CLS token)
        # -----------------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # -----------------------------------------------------------------
        # Forecasting head — placeholder for future extension.
        # Not enabled because the dataset lacks true longitudinal sequences.
        # When longitudinal multimodal data becomes available, replace this
        # with a GRU- or TransformerDecoder-based ForecastingHead.
        # -----------------------------------------------------------------
        if forecast_horizon > 0:
            self.forecast_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, forecast_horizon),
            )
        else:
            self.forecast_head = None

        # Initialize projection weights
        self._init_weights()

    # -----------------------------------------------------------------
    def _init_weights(self):
        """Initialize linear projection weights."""
        for proj in [self.vision_proj, self.audio_proj, self.physio_proj]:
            nn.init.trunc_normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        for module in self.cls_head:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    # -----------------------------------------------------------------
    def freeze_encoder_layer(self, layer_idx: int):
        """Freeze one encoder layer for ablation experiments.

        Args:
            layer_idx: 0 = first layer, -1 = last layer.
        """
        layer = self.encoder.layers[layer_idx]
        frozen = 0
        for param in layer.parameters():
            param.requires_grad = False
            frozen += param.numel()
        print(f"  Frozen encoder layer {layer_idx} ({frozen:,} params)")

    # -----------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: [B, 1024] concatenated features (vision_768 + audio_128 + physio_128).
            return_features: If True, return (logits, fusion_embedding).

        Returns:
            logits:           [B, num_classes] classification scores.
            fusion_embedding: [B, d_model] CLS token embedding (if return_features=True).
        """
        B = x.size(0)

        # Split concatenated input into modality features
        v = x[:, :768]                    # [B, 768]
        a = x[:, 768:896]                 # [B, 128]
        p = x[:, 896:]                    # [B, 128]

        # Project each modality → d_model
        v_tok = self.vision_proj(v)       # [B, d_model]
        a_tok = self.audio_proj(a)        # [B, d_model]
        p_tok = self.physio_proj(p)       # [B, d_model]

        # Stack tokens: [CLS, Vision, Audio, Physio]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls_tokens, v_tok.unsqueeze(1), a_tok.unsqueeze(1), p_tok.unsqueeze(1)], dim=1)
        # tokens: [B, 4, d_model]

        # Add positional embedding
        tokens = tokens + self.pos_embed   # [B, 4, d_model]

        # Transformer encoder with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            for layer in self.encoder.layers:
                tokens = _grad_checkpoint(layer, tokens, use_reentrant=False)
        else:
            tokens = self.encoder(tokens)  # [B, 4, d_model]

        # Extract CLS token → LayerNorm
        cls_embedding = self.norm(tokens[:, 0])  # [B, d_model]

        # Classification
        logits = self.cls_head(cls_embedding)     # [B, num_classes]

        if return_features:
            return logits, cls_embedding
        return logits

    # -----------------------------------------------------------------
    def get_features(self, x: Tensor) -> Tensor:
        """Extract fusion embedding (d_model dims) without gradient tracking."""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features

    # -----------------------------------------------------------------
    def count_parameters(self) -> dict:
        """Return a breakdown of trainable parameter counts."""
        def _count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "vision_proj": _count(self.vision_proj),
            "audio_proj": _count(self.audio_proj),
            "physio_proj": _count(self.physio_proj),
            "cls_token": self.cls_token.numel(),
            "pos_embed": self.pos_embed.numel(),
            "encoder": _count(self.encoder),
            "norm": _count(self.norm),
            "cls_head": _count(self.cls_head),
            "total": _count(self),
        }


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (tensor of shape [num_classes]).
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    vision_dim: int = 768,
    audio_dim: int = 128,
    physio_dim: int = 128,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    num_classes: int = 3,
    use_checkpoint: bool = False,
    forecast_horizon: int = 0,
) -> MultimodalFusionEncoder:
    """Factory function to create a MultimodalFusionEncoder.

    Args:
        forecast_horizon: Placeholder for future forecasting (0 = disabled).

    Returns:
        MultimodalFusionEncoder instance.
    """
    model = MultimodalFusionEncoder(
        vision_dim=vision_dim,
        audio_dim=audio_dim,
        physio_dim=physio_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        forecast_horizon=forecast_horizon,
    )
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model(num_classes=3, use_checkpoint=False)
    param_info = model.count_parameters()
    print("MultimodalFusionEncoder parameter breakdown:")
    for k, v in param_info.items():
        print(f"  {k}: {v:,}")

    # Test forward pass
    x = torch.randn(2, 1024)
    logits, feats = model(x, return_features=True)
    print(f"\nInput shape:   {x.shape}")
    print(f"Logits shape:  {logits.shape}")
    print(f"Features shape: {feats.shape}")

    # Test gradient checkpointing
    model_ckpt = build_model(num_classes=3, use_checkpoint=True)
    model_ckpt.train()
    logits2 = model_ckpt(x)
    print(f"\nCheckpointed model output: {logits2.shape}")

    # Test with binary classes
    model_bin = build_model(num_classes=2, use_checkpoint=False)
    logits3, feats3 = model_bin(x, return_features=True)
    print(f"\nBinary model — logits: {logits3.shape}, features: {feats3.shape}")

    # Test FocalLoss
    fl = FocalLoss(gamma=2.0)
    targets = torch.tensor([0, 2])
    loss = fl(logits, targets)
    print(f"\nFocalLoss: {loss.item():.4f}")
