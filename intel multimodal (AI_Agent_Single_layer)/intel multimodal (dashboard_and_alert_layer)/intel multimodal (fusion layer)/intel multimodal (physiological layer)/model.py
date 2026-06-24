"""
model.py — iTransformerClassifier: lightweight time-series model.

Architecture: 1D CNN projection + Transformer Encoder across variable tokens.

Input:  [B, L, C]  where L=1250 (time steps), C=4 (channels)
  1. Transpose → [B, C, L]
  2. seq_proj: Linear(L, d_model) — projects each variable's full time series
  3. + var_embedding: learned [1, C, d_model] positional embedding per variable
  4. TransformerEncoder: n_layers × (MHA + FFN), norm_first, GELU, batch_first
  5. Global average pool over C dimension → [B, d_model]
  6. LayerNorm → cls_head → logits

Matches architecture from IntelCup2026.ipynb cell 4 exactly.
Supports gradient checkpointing for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class iTransformerClassifier(nn.Module):
    """
    iTransformer-style classifier that applies Transformer attention across
    variable tokens (channels) rather than across time steps.

    Args:
        seq_len: Length of input time series (default 1250).
        n_vars: Number of input channels/variables (default 4).
        num_classes: Number of output classes (3 or 2).
        d_model: Model dimension / embedding size (default 128).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of Transformer encoder layers (default 3).
        d_ff: Feed-forward network hidden dimension (default 256).
        dropout: Dropout rate (default 0.1).
        use_checkpoint: If True, use gradient checkpointing on encoder layers
            to reduce memory usage at the cost of ~20% slower forward pass.
    """

    def __init__(
        self,
        seq_len: int = 1250,
        n_vars: int = 4,
        num_classes: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.use_checkpoint = use_checkpoint

        # 1. Project each variable's full time series into a d_model embedding
        self.seq_proj = nn.Linear(seq_len, d_model)

        # 2. Learnable per-variable embedding (like positional encoding for channels)
        self.var_embedding = nn.Parameter(torch.zeros(1, n_vars, d_model))
        nn.init.normal_(self.var_embedding, std=0.02)

        # 3. Transformer Encoder (attention across variable tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # 4. LayerNorm after pooling
        self.norm = nn.LayerNorm(d_model)

        # 5. Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, return_features: bool = False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, L, C] (batch, time, channels).
            return_features: If True, return (logits, features) where features
                is the 128-dim CLS embedding before the classification head.

        Returns:
            logits: [B, num_classes]
            OR (logits, features) if return_features=True.
        """
        B, L, C = x.shape

        # Transpose: [B, L, C] → [B, C, L] — each variable becomes a token
        h = x.transpose(1, 2)  # [B, C, L]

        # Project time series → d_model
        h = self.seq_proj(h)  # [B, C, d_model]

        # Add variable embedding
        h = h + self.var_embedding  # [B, C, d_model]

        # Transformer encoder across variable tokens
        if self.use_checkpoint and self.training:
            # Apply gradient checkpointing per layer for memory savings
            h = self._encoder_checkpointed(h)
        else:
            h = self.encoder(h)  # [B, C, d_model]

        # Global average pool over variable tokens
        h = h.mean(dim=1)  # [B, d_model]

        # LayerNorm
        h = self.norm(h)  # [B, d_model]

        # Classification head
        features = h  # CLS embedding (128-dim)
        logits = self.cls_head(features)  # [B, num_classes]

        if return_features:
            return logits, features
        return logits

    def _encoder_checkpointed(self, h):
        """Apply encoder layers one by one with gradient checkpointing."""
        for layer in self.encoder.layers:
            h = torch.utils.checkpoint.checkpoint(
                layer, h, use_reentrant=False
            )
        return h

    def get_features(self, x):
        """Extract CLS embedding (128-dim) without gradient tracking."""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features

    def count_parameters(self) -> dict:
        """Return a breakdown of trainable parameter counts."""
        def _count(m):
            if isinstance(m, nn.Parameter):
                return m.numel()
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "seq_proj": _count(self.seq_proj),
            "var_embedding": self.var_embedding.numel(),
            "encoder": _count(self.encoder),
            "norm": _count(self.norm),
            "cls_head": _count(self.cls_head),
            "total": _count(self),
        }


# ---------------------------------------------------------------------------
# Focal Loss (from notebook cell 4)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (tensor of shape [num_classes]).
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Convenience: build model with default config
# ---------------------------------------------------------------------------

def build_model(
    seq_len: int = 1250,
    n_vars: int = 4,
    num_classes: int = 3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    use_checkpoint: bool = False,
) -> iTransformerClassifier:
    """
    Factory function to create an iTransformerClassifier with default BIDMC config.

    Returns:
        iTransformerClassifier instance.
    """
    model = iTransformerClassifier(
        seq_len=seq_len,
        n_vars=n_vars,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    )
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model(num_classes=3, use_checkpoint=False)
    param_info = model.count_parameters()
    print(f"iTransformerClassifier parameter breakdown:")
    for k, v in param_info.items():
        print(f"  {k}: {v:,}")
    print(f"  Total: {param_info['total']:,}")

    # Test forward pass
    x = torch.randn(2, 1250, 4)
    logits = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test feature extraction
    logits2, feats = model(x, return_features=True)
    print(f"Features shape: {feats.shape}")

    # Test gradient checkpointing
    model_ckpt = build_model(num_classes=3, use_checkpoint=True)
    logits3 = model_ckpt(x)
    print(f"\nCheckpointed model output shape: {logits3.shape}")
    print(f"Checkpointed model params: {model_ckpt.count_parameters()['total']:,}")
