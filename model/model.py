"""
model/model.py

FashionClassifier — EfficientNet-B0 backbone with two independent
classification heads for gender and sleeve type prediction.

Architecture:
    Input (3 x 224 x 224)
        ↓
    EfficientNet-B0 backbone  (pretrained on ImageNet)
        ↓
    Shared feature vector  (1280-dim)
        ↓              ↓
    Gender head        Sleeve head
    Linear(1280 → 2)   Linear(1280 → 2)
        ↓              ↓
    male / female      full / half sleeve
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NUM_GENDER_CLASSES, NUM_SLEEVE_CLASSES, MODEL_NAME, MODEL_VERSION, CHECKPOINTS_DIR

log = logging.getLogger(__name__)

EFFICIENTNET_FEATURE_DIM = 1280  # output channels of EfficientNet-B0 backbone


class FashionClassifier(nn.Module):
    """
    Multi-task classifier built on EfficientNet-B0.
    Two output heads share the same backbone — the model learns
    general fashion features once and applies them to both tasks.
    """

    def __init__(
        self,
        num_gender_classes: int = NUM_GENDER_CLASSES,
        num_sleeve_classes: int = NUM_SLEEVE_CLASSES,
        dropout:            float = 0.3,
        pretrained:         bool  = True,
    ):
        super().__init__()

        weights       = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base          = models.efficientnet_b0(weights=weights)
        self.backbone = base.features                          # (B, 1280, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)               # (B, 1280, 1, 1)
        self.dropout  = nn.Dropout(p=dropout)
        self.gender_head = nn.Linear(EFFICIENTNET_FEATURE_DIM, num_gender_classes)
        self.sleeve_head = nn.Linear(EFFICIENTNET_FEATURE_DIM, num_sleeve_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:   x — image tensor (B, 3, 224, 224)
        Returns: gender_logits (B, 2), sleeve_logits (B, 2)
        """
        features = self.backbone(x)       # (B, 1280, 7, 7)
        features = self.pool(features)    # (B, 1280, 1, 1)
        features = features.flatten(1)    # (B, 1280)
        features = self.dropout(features)
        return self.gender_head(features), self.sleeve_head(features)

    # ── Freeze / unfreeze ─────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone weights — only heads will train."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info("Backbone frozen. Training heads only.")

    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Unfreeze the last N blocks of the EfficientNet backbone.
        EfficientNet-B0 has 9 feature blocks (indices 0–8).
        Called at the phase 1 → phase 2 transition in train.py.
        """
        for block in list(self.backbone.children())[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Unfroze last {n} backbone blocks. Trainable params: {trainable:,}")

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save(self, epoch: int, val_acc: float, path=None):
        """Save weights + metadata. Returns the saved path."""
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = CHECKPOINTS_DIR / f"{MODEL_NAME}_{MODEL_VERSION}_epoch{epoch:02d}_acc{val_acc:.3f}.pth"
        torch.save({
            "epoch":         epoch,
            "val_acc":       val_acc,
            "model_name":    MODEL_NAME,
            "model_version": MODEL_VERSION,
            "state_dict":    self.state_dict(),
        }, path)
        log.info(f"Checkpoint saved → {path}")
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    model = FashionClassifier(pretrained=False)

    total    = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())
    heads    = sum(p.numel() for p in model.gender_head.parameters()) + \
               sum(p.numel() for p in model.sleeve_head.parameters())
    print(f"backbone: {backbone:,}  heads: {heads:,}  total: {total:,}")

    dummy        = torch.zeros(2, 3, 224, 224)
    g_out, s_out = model(dummy)
    print(f"gender logits: {g_out.shape}  sleeve logits: {s_out.shape}")
    print("model.py OK")