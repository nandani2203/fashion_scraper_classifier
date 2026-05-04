"""
app/pages/products.py

Responsibility: show scraped products, label distribution, trigger label prep.
Imports from: config, data.label_prep
No session state. No custom HTML.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_CSV, LABELED_CSV, IMAGES_DIR


def render():
    st.header("Products")

    raw_df     = pd.read_csv(RAW_CSV)     if RAW_CSV.exists()     else None
    labeled_df = pd.read_csv(LABELED_CSV) if LABELED_CSV.exists() else None
    n_images   = len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0

    # ── row counts ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw products",    len(raw_df)     if raw_df     is not None else 0)
    c2.metric("Labeled samples", len(labeled_df) if labeled_df is not None else 0)
    c3.metric("Images on disk",  n_images)

    st.divider()

    # ── label prep trigger ────────────────────────────────────────────────
    if st.button("Run label prep"):
        from data.label_prep import prepare
        try:
            rows = prepare()
            st.success(f"{len(rows)} labeled rows saved to {LABELED_CSV.name}")
            st.rerun()
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))

    if labeled_df is None:
        st.info("No labeled data yet. Run the scraper first, then click 'Run label prep'.")
        st.code("python scraper/flipkart_scraper.py --pages 5", language="bash")
        return

    # ── distribution ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gender distribution")
        st.bar_chart(labeled_df["gender"].value_counts())
    with c2:
        st.subheader("Sleeve distribution")
        st.bar_chart(labeled_df["sleeve_type"].value_counts())

    # ── product table ─────────────────────────────────────────────────────
    st.subheader("Labeled products")
    st.dataframe(
        labeled_df[["title", "gender", "sleeve_type", "source"]].reset_index(drop=True),
        use_container_width=True,
    )