import json
from typing import Any

import time 
import numpy as np

import streamlit as st

from tqdm import tqdm
from loguru import logger

import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime, timedelta, UTC

from shapely.geometry import Point 
from folium import Map, CircleMarker
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


st.title("Hourly Trip Predictor for Chicago's Divvy Service")
current_hour = pd.to_datetime(datetime.now(UTC)).floor("H")
st.header(f"{current_hour} UTC")

displayed_scenario_names = {"start": "Departures", "end": "Arrivals"}

progress_bar = st.sidebar.header("⚙️ Working Progress")
progress_bar = st.sidebar.progress(value=0)
n_steps = 4

@st.cache_data
def load_geojson(scenario: str, indexer: str = "two") -> dict:
    """

    Args:
        scenario (str): _description_

    Returns:
        pd.DataFrame: 
        :param indexer:
    """
    with st.spinner(text="Getting the coordinates of each station..."):

        if indexer == "one":

            with open(GEOGRAPHICAL_DATA / f"rounded_{scenario}_points_and_new_ids.geojson") as file:
                points_and_ids = json.load(file)

            loaded_geodata = pd.DataFrame(
                {
                    f"{scenario}_station_id": points_and_ids.keys(), 
                    "coordinates": points_and_ids.values()
                }
            )

            reverse_geocoding = ReverseGeocoding(scenario=scenario, geodata=loaded_geodata)
            station_names_and_locations = reverse_geocoding.reverse_geocode()

            updated_geodata = reverse_geocoding.put_station_names_in_geodata(
                station_names_and_coordinates=station_names_and_locations
            )

            return updated_geodata
        
        elif indexer == "two":
            with open(INDEXER_TWO/f"{scenario}_geodata.geojson") as file:
                geodata_dict = json.load(file)

    st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")
    return geodata_dict


@st.cache_data
def get_features(scenario: str, target_date: datetime, local: bool = False) -> pd.DataFrame:
    """
    Initiate an inference object and use it to get features until the target date.
    features that we will use to fuel the model and produce predictions.

    Args:
        scenario (str): _description_
        target_date (datetime): _description_

    Returns:
        pd.DataFrame: the created (or fetched) features
    """
    with st.spinner(text="Getting a batch of features from the store..."):
        inferrer = InferenceModule(scenario=scenario)
        features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=False, local=local)
        
    return features 


@st.cache_data
def get_hourly_predictions(
    scenario: str,
    model_name: str, 
    from_hour: datetime = current_hour - timedelta(hours=1),
    to_hour: datetime = current_hour
) -> pd.DataFrame:
    """
    Initialise an inference object, and load a dataframe of predictions from a dedicated feature group
    on the offline feature store. We then fetch the most recent prediction if it is available, or the second
    most recent (the one from an hour before)

    Args:
        scenario (str): "start" for departures and "stop" for arrivals
        model_name (str): the name of the model to be used to perform the predictions
        from_hour (datetime): the starting ime from which we want to start making predictions
        to_hour (datetime): the hour with respect to which we want predictions. 

    Raises:
        Exception: In the event that the predictions for the current hour, or the previous one cannot be obtained.
                    This exception suggests that the feature pipeline may not be working properly.

    Returns:
        pd.DataFrame: dataframe containing hourly predicted arrivals or departures.
    """
    with st.spinner(text=f"Fetching predicted {displayed_scenario_names[scenario].lower()} from the feature store..."):

        inferrer = InferenceModule(scenario=scenario)

        predictions_df: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        next_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == to_hour]

        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == from_hour]
            current_hour = from_hour

        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

    return predictions_to_use


#@st.cache_data
def make_geodataframe(geojson: dict, scenario: str) -> GeoDataFrame:
    
    coordinates = []
    station_ids = []
    station_names = []
    
    for detail_index in range(len(geojson["features"])):
            
        detail: dict = geojson["features"][detail_index]

        coordinates.append(
            detail["geometry"]["coordinate"]
        )

        station_ids.append(
            detail["properties"]["station_id"]
        )

        station_names.append(
            detail["properties"]["station_name"]
        )
    
    latitudes = [point[1] for point in coordinates]
    longitudes = [point[0] for point in coordinates]

    data = pd.DataFrame(
        data={
            f"{scenario}_station_name": station_names,
            f"{scenario}_station_id": station_ids,
            "latitudes": latitudes,
            "longitudes": longitudes
        }
    )

    #  data["coordinates"] = data["coordinates"].apply(lambda x: Point(x[0], x[1]))
    #  geodata = GeoDataFrame(data=data, geometry="coordinates")

    #  geodata.set_crs(epsg=4326, inplace=True)

    progress_bar.progress(1 / n_steps)
    return data


def make_map(scenario: str, geodata: GeoDataFrame, predictions: pd.DataFrame):
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
        longitudes = geodata["longitudes"].values

        #rows_to_iterate = tqdm(iterable=range(geodata.shape[0]), desc="Gathering elements to display")
        #predictions_per_id = predictions.set_index(f"{scenario}_station_id")[f"predicted_{scenario}s"]

        marker_cluster = FastMarkerCluster(data=zip(latitudes, longitudes), popups=f"{predictions_per_id.values}")
        folium_map = Map(location=centre)
        marker_cluster.add_to(parent=folium_map)

        displayed_map = st_folium(fig=folium_map, width=900, height=650)

    logger.success("Displayed map")
    st.sidebar.write("✅ Map Drawn")
    progress_bar.progress(4/n_steps)


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

    for scenario in displayed_scenario_names.keys():
        if displayed_scenario_names[scenario] in user_scenario_choice:

            # Prepare geodata
            geojson = load_geojson(scenario=scenario)
            geodata = make_geodataframe(geojson=geojson, scenario=scenario)
            
            # Fetch features and predictions<
            features = get_features(scenario=scenario, target_date=current_hour, local=True)
            st.sidebar.write("✅ Fetched features for inference")
            progress_bar.progress(2/n_steps)

            predictions = get_hourly_predictions(scenario=scenario, model_name=model_name)

            if not predictions.empty:
                st.sidebar.write("✅ Model's predictions received")
                progress_bar.progress(3 / n_steps)

            make_map(scenario=scenario, geodata=geodata, predictions=predictions)
        

if __name__ == "__main__":
    construct_page(model_name="lightgbm")
