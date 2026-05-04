"""
app/pages/predict_page.py

Responsibility: single and batch prediction UI.
Imports from: app.state (get_predictor), app.components (display),
              database.db (save results), inference.predict (Predictor)
No custom HTML. No session state writes (delegated to state.py).
"""

import sys
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))


def render():
    st.header("Predict")

    from app.state import get_predictor
    from app.components import show_prediction_result, show_results_table, show_no_model_warning

    predictor = get_predictor()
    if predictor is None:
        show_no_model_warning()
        return

    st.caption(f"Model: {predictor.model_name} {predictor.model_version}")
    tab_single, tab_batch = st.tabs(["Single image", "Batch"])

    # ── Single ────────────────────────────────────────────────────────────
    with tab_single:
        st.subheader("Single image prediction")
        url = st.text_input("Image URL or local file path")

        if url:
            st.image(url, width=300)

        if url and st.button("Predict", key="btn_single"):
            with st.spinner("Running…"):
                result = predictor.predict_single(url)

            from database.db import save_prediction
            save_prediction(result)

            show_prediction_result(result)

    # ── Batch ─────────────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch prediction")

        input_method = st.radio(
            "Input method", ["Paste URLs", "Upload CSV"], horizontal=True
        )

        urls = []
        if input_method == "Paste URLs":
            raw = st.text_area(
                "One image URL per line",
                height=150,
                placeholder="https://example.com/img1.jpg\nhttps://example.com/img2.jpg",
            )
            if raw.strip():
                urls = [u.strip() for u in raw.splitlines() if u.strip()]
        else:
            uploaded = st.file_uploader("CSV with image_url column", type=["csv"])
            if uploaded:
                df_up    = pd.read_csv(uploaded)
                col_name = "image_url" if "image_url" in df_up.columns else df_up.columns[0]
                urls     = df_up[col_name].dropna().tolist()
                st.caption(f"{len(urls)} URLs found in column '{col_name}'")

        if urls:
            st.caption(f"{len(urls)} image(s) queued")

        if urls and st.button("Run batch", key="btn_batch", type="primary"):
            with st.spinner(f"Predicting {len(urls)} images…"):
                results = predictor.predict_batch(urls)

            from database.db import save_predictions
            save_predictions(results)

            success = sum(1 for r in results if r.status == "success")
            st.success(f"{success}/{len(results)} predictions complete — saved to history.")
            show_results_table(results)