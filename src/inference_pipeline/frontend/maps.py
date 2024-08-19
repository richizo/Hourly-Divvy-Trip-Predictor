import numpy as np
import pandas as pd

import streamlit as st

from tqdm import tqdm
from loguru import logger

from geopandas import GeoDataFrame
from shapely.geometry import Point 
from folium import Map, CircleMarker
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from src.setup.config import config, choose_displayed_scenario_name
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO

from src.inference_pipeline.frontend.predictions import (
    tracker, load_geojson, make_geodataframe, get_features, get_hourly_predictions
)


def make_scatterplot(geodata: GeoDataFrame, predictions: pd.DataFrame):
    """
    
    Args:
        scenario (str): _description_
        geojson (dict): _description_
        geodata (GeoDataFrame): _description_
        predictions (pd.DataFrame): _description_
    """
    with st.spinner("Building map..."):
        centre = [41.872866, -87.63363]
        station_names = geodata.iloc[:, 0].values
        station_ids = geodata.iloc[:, 1].values

        latitudes = geodata["latitudes"].values
        print(len(latitudes))
        longitudes = geodata["longitudes"].values

        #rows_to_iterate = tqdm(iterable=range(geodata.shape[0]), desc="Gathering elements to display")
        #predictions_per_id = predictions.set_index(f"{scenario}_station_id")[f"predicted_{scenario}s"]

        marker_cluster = FastMarkerCluster(data=zip(latitudes, longitudes))
        folium_map = Map(location=centre)
        marker_cluster.add_to(parent=folium_map)

        displayed_map = st_folium(fig=folium_map, width=900, height=650)

    st.sidebar.write("✅ Map Drawn")
    tracker.next()
    

user_scenario_choice: list[str] = st.sidebar.multiselect(
    label="Do you want predictions for the number of arrivals at or the departures from each station?",
    options=["Arrivals", "Departures"],
    placeholder="Please select one of the two options."
)

for scenario in ["start", "end"]:
    displayed_scenario_names = choose_displayed_scenario_name()
    if displayed_scenario_names[scenario] in user_scenario_choice:

        # Prepare geodata
        geojson = load_geojson(scenario=scenario, indexer="two")
        geodata = make_geodataframe(scenario=scenario, geojson=geojson)
        tracker.next()
        
        # Fetch features and predictions<
        features = get_features(scenario=scenario, target_date=config.current_hour)
        predictions = get_hourly_predictions(scenario=scenario)

        make_scatterplot(geodata=geodata, predictions=predictions)
