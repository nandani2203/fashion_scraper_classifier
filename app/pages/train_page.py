"""
app/pages/train_page.py

Responsibility: configure hyperparameters, trigger training, show results.
Imports from: config, model.train, app.state (to clear predictor cache)
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    LABELED_CSV, CHECKPOINTS_DIR,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, FREEZE_EPOCHS,
    GENDER_CLASSES, SLEEVE_CLASSES,
)


def render():
    st.header("Train")

    # ── existing checkpoints ──────────────────────────────────────────────
    ckpts = sorted(CHECKPOINTS_DIR.glob("*.pth"), reverse=True)
    if ckpts:
        st.subheader("Saved checkpoints")
        import torch
        rows = []
        for p in ckpts:
            try:
                meta = torch.load(p, map_location="cpu")
                rows.append({
                    "file":    p.name,
                    "epoch":   meta.get("epoch", "?"),
                    "val_acc": f"{meta.get('val_acc', 0):.3f}",
                    "version": meta.get("model_version", "?"),
                })
            except Exception:
                rows.append({"file": p.name, "epoch": "?", "val_acc": "?", "version": "?"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.divider()

    # ── data check ────────────────────────────────────────────────────────
    if not LABELED_CSV.exists():
        st.warning("No labeled data found. Go to **Products** and run label prep first.")
        return

    labeled_count = sum(1 for _ in open(LABELED_CSV)) - 1
    st.info(f"{labeled_count} labeled samples ready for training.")

    # ── hyperparameters ───────────────────────────────────────────────────
    st.subheader("Hyperparameters")
    c1, c2 = st.columns(2)
    with c1:
        epochs    = st.slider("Epochs", 5, 50, NUM_EPOCHS)
        batch     = st.select_slider("Batch size", options=[8, 16, 32, 64], value=BATCH_SIZE)
    with c2:
        lr        = st.select_slider(
                        "Learning rate",
                        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                        value=LEARNING_RATE,
                        format_func=lambda x: f"{x:.0e}",
                    )
        freeze_ep = st.slider("Freeze epochs (phase 1)", 1, 10, FREEZE_EPOCHS)

    device_opt = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"])
    device_arg = None if device_opt == "auto" else device_opt

    st.divider()

    if st.button("Start training", type="primary"):
        from model.train import train
        from app.state import clear_predictor

        with st.spinner(f"Training for {epochs} epochs…"):
            try:
                best_path, history, val_metrics = train(
                    num_epochs    = epochs,
                    batch_size    = batch,
                    learning_rate = lr,
                    freeze_epochs = freeze_ep,
                    device_name   = device_arg,
                )
                clear_predictor()
                st.success(f"Training complete. Best checkpoint: {Path(best_path).name}")

                # ── accuracy chart ────────────────────────────────────────
                if history:
                    df = pd.DataFrame(history).set_index("epoch")
                    st.subheader("Validation accuracy")
                    st.line_chart(df[["val_gender_acc", "val_sleeve_acc", "val_combined"]])

                # ── classification reports ────────────────────────────────
                st.subheader("Model validation")
                st.caption(
                    "Accuracy alone is misleading on imbalanced data. "
                    "Precision, recall, and F1 show per-class performance."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Gender**")
                    g_report = val_metrics["gender_report"]
                    g_rows   = [
                        {
                            "class":     cls,
                            "precision": f"{g_report[cls]['precision']:.2f}",
                            "recall":    f"{g_report[cls]['recall']:.2f}",
                            "f1":        f"{g_report[cls]['f1-score']:.2f}",
                            "support":   g_report[cls]["support"],
                        }
                        for cls in GENDER_CLASSES if cls in g_report
                    ]
                    st.dataframe(pd.DataFrame(g_rows), hide_index=True, use_container_width=True)

                    # confusion matrix
                    st.markdown("Confusion matrix")
                    g_cm = val_metrics["gender_cm"]
                    st.dataframe(
                        pd.DataFrame(g_cm, index=GENDER_CLASSES, columns=GENDER_CLASSES),
                        use_container_width=True,
                    )

                with c2:
                    st.markdown("**Sleeve type**")
                    s_report = val_metrics["sleeve_report"]
                    s_rows   = [
                        {
                            "class":     cls,
                            "precision": f"{s_report[cls]['precision']:.2f}",
                            "recall":    f"{s_report[cls]['recall']:.2f}",
                            "f1":        f"{s_report[cls]['f1-score']:.2f}",
                            "support":   s_report[cls]["support"],
                        }
                        for cls in SLEEVE_CLASSES if cls in s_report
                    ]
                    st.dataframe(pd.DataFrame(s_rows), hide_index=True, use_container_width=True)

                    st.markdown("Confusion matrix")
                    s_cm = val_metrics["sleeve_cm"]
                    st.dataframe(
                        pd.DataFrame(s_cm, index=SLEEVE_CLASSES, columns=SLEEVE_CLASSES),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Training failed: {e}")