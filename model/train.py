"""
model/train.py

Two-phase training loop for FashionClassifier.

Phase 1 (epochs 1 → freeze_epochs):
    Backbone frozen. Only the two heads train.
    High learning rate — heads start from random weights.

Phase 2 (epochs freeze_epochs+1 → num_epochs):
    Last 3 backbone blocks unfrozen + heads.
    Lower learning rate — fine-tune deep features for fashion.

Logs per-epoch: loss, gender_acc, sleeve_acc, combined_acc.
Saves best checkpoint by combined val accuracy.
After training: prints confusion matrix + classification report per task.
"""

import logging
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FREEZE_EPOCHS, LABELED_CSV, GENDER_CLASSES, SLEEVE_CLASSES,
)
from data.dataset import get_dataloaders
from model.model import FashionClassifier

log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────

class RunningMetrics:
    """Accumulates loss and accuracy over one epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss     = 0.0
        self.gender_correct = 0
        self.sleeve_correct = 0
        self.n_samples      = 0

    def update(self, loss, gender_logits, sleeve_logits, gender_labels, sleeve_labels):
        n = gender_labels.size(0)
        self.total_loss     += loss * n
        self.gender_correct += (gender_logits.argmax(1) == gender_labels).sum().item()
        self.sleeve_correct += (sleeve_logits.argmax(1) == sleeve_labels).sum().item()
        self.n_samples      += n

    @property
    def avg_loss(self)     -> float: return self.total_loss     / max(self.n_samples, 1)
    @property
    def gender_acc(self)   -> float: return self.gender_correct / max(self.n_samples, 1)
    @property
    def sleeve_acc(self)   -> float: return self.sleeve_correct / max(self.n_samples, 1)
    @property
    def combined_acc(self) -> float: return (self.gender_acc + self.sleeve_acc) / 2


# ── Single epoch ──────────────────────────────────────────────────────────

def run_epoch(model, loader, gender_criterion, sleeve_criterion, optimizer, device, train: bool) -> RunningMetrics:
    """Run one train or validation epoch. Returns accumulated metrics."""
    model.train(train)
    metrics = RunningMetrics()

    with torch.set_grad_enabled(train):
        for images, gender_labels, sleeve_labels in loader:
            images        = images.to(device)
            gender_labels = gender_labels.to(device)
            sleeve_labels = sleeve_labels.to(device)

            gender_logits, sleeve_logits = model(images)
            loss_gender = gender_criterion(gender_logits, gender_labels)
            loss_sleeve = sleeve_criterion(sleeve_logits, sleeve_labels)

            loss = loss_gender + loss_sleeve

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics.update(
                loss.item(),
                gender_logits.detach(), sleeve_logits.detach(),
                gender_labels, sleeve_labels,
            )

    return metrics


# ── Model validation ──────────────────────────────────────────────────────

def evaluate(model, val_loader, device) -> dict:
    """
    Run the trained model over the validation set and return:
      - classification report (precision, recall, F1) for gender and sleeve
      - confusion matrix for both tasks
      - raw prediction lists for any further analysis

    Called once after the training loop completes.
    Accuracy alone is misleading on imbalanced datasets —
    this shows per-class breakdown so you can see exactly where the model fails.
    """
    model.eval()
    all_g_pred, all_g_true = [], []
    all_s_pred, all_s_true = [], []

    with torch.no_grad():
        for images, g_labels, s_labels in val_loader:
            g_logits, s_logits = model(images.to(device))
            all_g_pred.extend(g_logits.argmax(1).cpu().tolist())
            all_g_true.extend(g_labels.tolist())
            all_s_pred.extend(s_logits.argmax(1).cpu().tolist())
            all_s_true.extend(s_labels.tolist())

    gender_report = classification_report(
        all_g_true, all_g_pred,
        target_names=GENDER_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    sleeve_report = classification_report(
        all_s_true, all_s_pred,
        target_names=SLEEVE_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    gender_cm = confusion_matrix(all_g_true, all_g_pred).tolist()
    sleeve_cm = confusion_matrix(all_s_true, all_s_pred).tolist()

    # log human-readable version
    log.info("── Gender classification report ──────────────────")
    log.info("\n" + classification_report(
        all_g_true, all_g_pred, target_names=GENDER_CLASSES, zero_division=0
    ))
    log.info("── Sleeve classification report ──────────────────")
    log.info("\n" + classification_report(
        all_s_true, all_s_pred, target_names=SLEEVE_CLASSES, zero_division=0
    ))
    log.info(f"Gender confusion matrix:\n{gender_cm}")
    log.info(f"Sleeve confusion matrix:\n{sleeve_cm}")

    return {
        "gender_report": gender_report,
        "sleeve_report": sleeve_report,
        "gender_cm":     gender_cm,
        "sleeve_cm":     sleeve_cm,
    }


# ── Training loop ─────────────────────────────────────────────────────────

def train(
    num_epochs:    int   = NUM_EPOCHS,
    batch_size:    int   = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay:  float = WEIGHT_DECAY,
    freeze_epochs: int   = FREEZE_EPOCHS,
    csv_path             = LABELED_CSV,
    device_name:   str   = None,
):
    device = torch.device(
        device_name if device_name else
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device} | epochs: {num_epochs} | batch: {batch_size}")
    log.info(f"Phase 1 (frozen backbone): epochs 1–{freeze_epochs}")
    log.info(f"Phase 2 (fine-tune):       epochs {freeze_epochs+1}–{num_epochs}")

    train_loader, val_loader, test_loader  = get_dataloaders(csv_path=csv_path, batch_size=batch_size)

    model = FashionClassifier(pretrained=True).to(device)
    model.freeze_backbone()

    gender_weights = torch.tensor([1.3, 1.0]).to(device)  # boost male (index 0)
    gender_criterion = nn.CrossEntropyLoss(weight=gender_weights)

    sleeve_weights = torch.tensor([1.5, 1.0]).to(device)  # boost full_sleeve (index 0)
    sleeve_criterion = nn.CrossEntropyLoss(weight=sleeve_weights)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc   = 0.0
    best_ckpt_path = None
    history        = []
    patience = 3
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if epoch == freeze_epochs + 1:
            log.info("Phase 2: unfreezing last 3 backbone blocks")
            model.unfreeze_last_n_blocks(n=3)
            optimizer = Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate * 0.1, weight_decay=weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=num_epochs - freeze_epochs, eta_min=1e-7
            )

        train_m = run_epoch(model, train_loader, gender_criterion,sleeve_criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   gender_criterion, sleeve_criterion, optimizer, device, train=False)
        scheduler.step()

        log.info(
            f"Epoch {epoch:02d}/{num_epochs} [{time.time()-t0:.1f}s]  "
            f"train_loss={train_m.avg_loss:.4f}  val_loss={val_m.avg_loss:.4f}  "
            f"val_gender={val_m.gender_acc:.3f}  val_sleeve={val_m.sleeve_acc:.3f}  "
            f"val_combined={val_m.combined_acc:.3f}"
        )

        history.append({
            "epoch":          epoch,
            "train_loss":     train_m.avg_loss,
            "val_loss":       val_m.avg_loss,
            "val_gender_acc": val_m.gender_acc,
            "val_sleeve_acc": val_m.sleeve_acc,
            "val_combined":   val_m.combined_acc,
        })

        if val_m.combined_acc > best_val_acc:
            best_val_acc = val_m.combined_acc
            best_ckpt_path = model.save(epoch, val_m.combined_acc)
            log.info(f"  ★ New best — combined_acc={best_val_acc:.3f}")
            
            no_improve_epochs = 0   # reset
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Training complete. Best combined_acc={best_val_acc:.3f} → {best_ckpt_path}")

    # ── post-training evaluation ──────────────────────────────────────────
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    return best_ckpt_path, history, val_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Train FashionClassifier")
    parser.add_argument("--epochs", type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze", type=int,   default=FREEZE_EPOCHS)
    parser.add_argument("--device", type=str,   default=None)
    parser.add_argument("--csv",    type=str,   default=str(LABELED_CSV))
    args = parser.parse_args()

    best_path, history, val_metrics, test_metrics  = train(
        num_epochs    = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        freeze_epochs = args.freeze,
        device_name   = args.device,
        csv_path      = Path(args.csv),
    )

    print(f"\nBest checkpoint: {best_path}")
    print(f"\n{'epoch':>5}  {'val_gender':>10}  {'val_sleeve':>10}  {'combined':>10}")
    for h in history:
        print(f"  {h['epoch']:>3}  {h['val_gender_acc']:>10.3f}  "
              f"{h['val_sleeve_acc']:>10.3f}  {h['val_combined']:>10.3f}")