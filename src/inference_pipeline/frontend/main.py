"""
This module initiates the streamlit app.
"""
import streamlit as st
from streamlit_extras.app_logo import add_logo

from src.setup.paths import IMAGES_DIR


add_logo(logo_url=IMAGES_DIR/"logo.png", height=120)

pages = st.navigation(
    pages=[
        st.Page(page="intro.py", title="Welcome", icon="🏠"), 
        st.Page(page="predictions.py", title="Predictions", icon="🔮"),
        #st.Page(page="plots.py", title="Viewing Trips Over Time", icon="📈"),
        # st.Page(page="monitoring.py", title="Monitoring Model Performance", icon="🔬"),
        st.Page(page="about.py", title="About Me", icon="🧔‍♂️")
    ]
)


pages.run()
