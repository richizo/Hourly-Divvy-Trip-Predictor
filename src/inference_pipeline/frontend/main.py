"""
This module initiates the streamlit app.
"""
import streamlit as st
from streamlit_extras.app_logo import add_logo

from src.setup.paths import IMAGES_DIR


class ProgressTracker:
    """
    A way for me to more conveniently advance the various progress bars that I will have 
    in the sidebar.
    """
    def __init__(self, n_steps: int):
        self.current_step = 0
        self.n_steps = n_steps
        self.progress_bar = st.sidebar.header("⚙️ Working Progress")
        self.progress_bar = st.sidebar.progress(value=0)

    def next(self) -> None:
        self.current_step += 1 
        self.progress_bar.progress(self.current_step/self.n_steps)


add_logo(logo_url=IMAGES_DIR/"logo.png", height=120)

pages = st.navigation(
    pages=[
        st.Page(page="intro.py", title="Welcome", icon="🏠"), 
        st.Page(page="predictions.py", title="Predictions", icon="👁️"),
        st.Page(page="maps.py", title="Map", icon="🗺️"),
        st.Page(page="plots.py", title="Plots of Trips Over Time", icon="📈"),
        st.Page(page="monitoring.py", title="Model Performance", icon="📈")
    ]
)

pages.run()
