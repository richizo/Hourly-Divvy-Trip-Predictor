"""
Contains a class that gives me way for me to more conveniently advance 
the various progress bars that I will have in the sidebar.

rKeeping this object in main.py (its original location) causes the widgets in 
whichever script calls the tracker to appear twice, so it needs 
to live in a dedicated file.
"""
import streamlit as st 

class ProgressTracker:
    def __init__(self, n_steps: int) -> None:
        self.current_step = 0
        self.n_steps = n_steps
        self.header = st.sidebar.header("⚙️ Working Progress")
        self.progress_bar = st.sidebar.progress(value=0)

    def next(self) -> None:
        self.current_step += 1 
        self.progress_bar.progress(self.current_step/self.n_steps)
