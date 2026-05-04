"""
app/pages/history.py

Responsibility: show prediction history, stats, filters, CSV export.
Imports from: database.db (get_history, get_stats, get_batch)
"""

import sys
from io import StringIO
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))


def render():
    st.header("History")

    from database.db import get_history, get_stats, get_batch

    stats = get_stats()

    # ── summary metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", stats["total"])
    c2.metric("Single runs",       stats["total_single"])
    c3.metric("Batch runs",        stats["total_batch"])
    c4.metric("Errors",            stats["total_errors"])

    if stats["total"] == 0:
        st.info("No predictions recorded yet. Go to **Predict** to run some.")
        return

    st.divider()

    # ── filters ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        rt_filter = st.selectbox("Run type", ["all", "single", "batch"])
    with c2:
        st_filter = st.selectbox("Status",   ["all", "success", "error"])
    with c3:
        limit = st.slider("Max rows", 10, 500, 100)

    history = get_history(
        limit    = limit,
        run_type = None if rt_filter == "all" else rt_filter,
        status   = None if st_filter == "all" else st_filter,
    )

    if not history:
        st.info("No records match the current filters.")
        return

    df = pd.DataFrame(history)
    st.caption(f"{len(df)} records")

    show_cols = [
        "id", "run_type", "predicted_gender", "predicted_sleeve",
        "confidence_gender", "confidence_sleeve", "model_version",
        "status", "timestamp",
    ]
    st.dataframe(df[[c for c in show_cols if c in df.columns]], use_container_width=True, hide_index=True)

    # ── batch inspector ───────────────────────────────────────────────────
    st.divider()
    batch_id = st.text_input("Inspect a batch — paste run_id")
    if batch_id:
        rows = get_batch(batch_id.strip())
        if rows:
            st.caption(f"{len(rows)} items in this batch")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No batch found with that run_id.")

    # ── export ────────────────────────────────────────────────────────────
    st.divider()
    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label     = "Export to CSV",
        data      = buf.getvalue(),
        file_name = "prediction_history.csv",
        mime      = "text/csv",
    )