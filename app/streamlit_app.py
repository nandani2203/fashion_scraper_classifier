"""
app/streamlit_app.py

Entry point. Responsibilities:
  - Page config
  - Sidebar navigation
  - Route to the correct page module
Nothing else.
"""

import sys
from pathlib import Path
import importlib

import streamlit as st


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.state import init_db

st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="wide",
)

st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

init_db()

PAGES = {
    "Products": "app.pages.products",
    "Train":    "app.pages.train_page",
    "Predict":  "app.pages.predict_page",
    "History":  "app.pages.history",
}

with st.sidebar:
    st.title("👗 Fashion Classifier")
    st.caption("EfficientNet-B0 · Gender + Sleeve")
    st.divider()
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

page = importlib.import_module(PAGES[choice])
page.render()