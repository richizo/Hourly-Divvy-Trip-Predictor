"""
This module initiates the streamlit app.
"""
import streamlit as st


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
        

intro_page = st.Page(page="intro.py", title="Welcome", icon="🏠")
predictions_page = st.Page(page="predictions.py", title="Predictions", icon="👁️")
# monitoring_page = st.Page(page="monitoring.py", title="Model Performance", icon="📈")
# maps_page = st.Page(page="maps.py", title="Maps (Experimental)", icon="🗺️")


pages = st.navigation(pages=[intro_page, predictions_page])
pages.run()
