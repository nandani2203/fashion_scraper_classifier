"""
app/components.py

Shared UI components used across multiple pages.
Rules:
  - Native Streamlit widgets only — no custom HTML or CSS
  - No imports from backend modules
  - No session state access
"""

import streamlit as st
import pandas as pd


def show_prediction_result(result):
    """
    Display a single PredictionResult using native Streamlit widgets.
    Used by both predict.py and history.py.
    """
    if result.status == "error":
        st.error(f"Prediction failed: {result.error_message}")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Gender",     result.predicted_gender)
        st.metric("Confidence", f"{result.confidence_gender:.1%}")
    with c2:
        st.metric("Sleeve type", result.predicted_sleeve.replace("_", " "))
        st.metric("Confidence",  f"{result.confidence_sleeve:.1%}")

    st.caption(
        f"run_id: {result.run_id[:18]}…  |  "
        f"model: {result.model_name} {result.model_version}"
    )


def show_results_table(results: list):
    """
    Display a list of PredictionResults as a clean dataframe.
    Used by predict.py after batch runs.
    """
    rows = [
        {
            "image_url":  r.image_url[:60] + "…" if len(r.image_url) > 60 else r.image_url,
            "gender":     r.predicted_gender,
            "sleeve":     r.predicted_sleeve.replace("_", " "),
            "conf_gender": f"{r.confidence_gender:.1%}",
            "conf_sleeve": f"{r.confidence_sleeve:.1%}",
            "status":     r.status,
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_no_model_warning():
    """Consistent warning shown on any page that needs a trained model."""
    st.warning("No trained model found. Go to **Train** and run training first.")