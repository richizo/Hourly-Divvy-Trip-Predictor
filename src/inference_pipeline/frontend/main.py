import streamlit as st

from datetime import UTC, datetime, timedelta

from predictions import ProgressTracker
from maps import make_scatterplot

from src.setup.config import config, choose_displayed_scenario_name
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_TWO, INDEXER_TWO


intro_page = st.Page(page="intro.py", title="Welcome", icon="🏠")
maps_page = st.Page(page="maps.py", title="Maps", icon="🗺️")
predictions_page = st.Page(page="predictions.py", title="Trip Predictions", icon="👁️")

pages = st.navigation(pages=[intro_page, maps_page, predictions_page])
pages.run()
