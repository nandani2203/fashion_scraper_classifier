"""
app/state.py

Single owner of st.session_state for the entire app.
Every piece of cached state is initialised and accessed here.
No other file touches st.session_state directly.
"""

import sys
from pathlib import Path

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINTS_DIR


def get_predictor():
    """
    Load the Predictor once and keep it in session state.
    Returns None if no checkpoint exists yet.
    """
    if "predictor" in st.session_state:
        return st.session_state.predictor

    ckpts = list(CHECKPOINTS_DIR.glob("*.pth"))
    if not ckpts:
        return None

    from inference.predict import Predictor
    with st.spinner("Loading model…"):
        st.session_state.predictor = Predictor(device="cpu")

    return st.session_state.predictor


def clear_predictor():
    """
    Call this after training completes so the next prediction
    reloads the freshly saved checkpoint.
    """
    st.session_state.pop("predictor", None)


def init_db():
    """Initialise the database once per session."""
    if "db_ready" not in st.session_state:
        from database.db import init_db as _init_db
        _init_db()
        st.session_state.db_ready = True