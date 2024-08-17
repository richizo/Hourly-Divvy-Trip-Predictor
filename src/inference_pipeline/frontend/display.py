import json
import streamlit as st

import pandas as pd
from datetime import datetime, timedelta, UTC

from predictions import ProgressTracker, Loader
from maps import MapMaker

from src.setup.config import config, choose_displayed_scenario_name
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


intro_page = st.Page(page="intro.py", title="Welcome", icon="🏠")

maps_page = st.Page(page="maps.py", title="Maps", icon="🗺️")
predictions_page = st.Page(page="predictions.py", title="Hourly Predictions Per Station", icon="👁️")

pages = st.navigation(pages=[intro_page, maps_page, predictions_page])
pages.run()


def construct_page(model_name: str):
    """


    Args:
        model_name (str): _description_
    """
    user_scenario_choice: list[str] = st.sidebar.multiselect(
        label="Do you want predictions for the number of arrivals at or the departures from each station?",
        options=["Arrivals", "Departures"],
        placeholder="Please select one of the two options."
    )

    progress_tracker = ProgressTracker(n_steps=4)

    for scenario in ["start", "end"]:
        displayed_scenario_names = choose_displayed_scenario_name()
        if displayed_scenario_names[scenario] in user_scenario_choice:

            # Prepare geodata
            geojson = load_geojson(scenario=scenario)
            geodata = make_geodataframe(geojson=geojson, scenario=scenario)
            
            # Fetch features and predictions<
            features = get_features(scenario=scenario, target_date=current_hour,     local=True)
            st.sidebar.write("✅ Fetched features for inference")
            progress_tracker.next()

            predictions = get_hourly_predictions(scenario=scenario, model_name=model_name)

            if not predictions.empty:
                st.sidebar.write("✅ Model's predictions received")
                progress_tracker.next()

            make_map(scenario=scenario, geodata=geodata, predictions=predictions)
        

if __name__ == "__main__":
    construct_page(model_name="lightgbm")
