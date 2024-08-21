import streamlit as st


intro_page = st.Page(page="intro.py", title="Welcome", icon="🏠")
maps_page = st.Page(page="maps.py", title="Maps", icon="🗺️")
predictions_page = st.Page(page="predictions.py", title="Trip Predictions", icon="👁️")

pages = st.navigation(pages=[intro_page, maps_page, predictions_page])
pages.run()
