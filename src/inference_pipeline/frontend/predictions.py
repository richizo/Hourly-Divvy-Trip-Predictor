import json
from typing import Any

import numpy as np
import streamlit as st

from tqdm import tqdm
from loguru import logger

import pandas as pd
from geopandas import GeoDataFrame
from datetime import UTC, datetime, timedelta

from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry

from src.setup.config import choose_displayed_scenario_name, config 
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


class ProgressTracker:

    def __init__(self, n_steps: int):
        
        self.current_step = 1
        self.n_steps = n_steps
        self.progress_bar = st.sidebar.header("⚙️ Working Progress")
        self.progress_bar = st.sidebar.progress(value=0)

    def next(self) -> None:
        self.progress_bar.progress(self.current_step/self.n_steps)
        self.current_step += 1 


displayed_scenario_names = choose_displayed_scenario_name()
tracker = ProgressTracker(n_steps=4)


@st.cache_data
def load_geojson(scenario: str, indexer: str) -> dict:
    """

    Args:
        indexer (str, optional): _description_. Defaults to "two".

    Returns:
        dict: _description_
    """
    with st.spinner(text="Getting the coordinates of each station..."):

        if indexer == "one":

            with open(GEOGRAPHICAL_DATA / f"rounded_{scenario}_points_and_new_ids.geojson") as file:
                points_and_ids = json.load(file)

            loaded_geodata = pd.DataFrame(
                {
                    f"{scenario}_station_id": points_and_ids.keys(), "coordinates": points_and_ids.values()
                }
            )

            reverse_geocoding = ReverseGeocoding(scenario=scenario, geodata=loaded_geodata)
            station_names_and_locations = reverse_geocoding.reverse_geocode()

            updated_geodata = reverse_geocoding.put_station_names_in_geodata(
                station_names_and_coordinates=station_names_and_locations
            )

            return updated_geodata
        
        elif indexer == "two":
            with open(INDEXER_TWO/f"{scenario}_geojson.geojson") as file:
                geodata_dict = json.load(file)
            return geodata_dict

    st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")
    tracker.next()

@st.cache_data
def get_features(scenario: str, target_date: datetime, geocode: bool = False) -> pd.DataFrame:
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
        features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=geocode)

    st.sidebar.write("✅ Fetched features for inference")
    tracker.next()
    return features 


@st.cache_data
def get_hourly_predictions(
    scenario: str,
    model_name: str = "lightgbm", 
    from_hour: datetime = config.current_hour - timedelta(hours=1),
    to_hour: datetime = config.current_hour
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
    with st.spinner(
        text=f"Fetching predicted {displayed_scenario_names[scenario].lower()} from the feature store..."
    ):
        inferrer = InferenceModule(scenario=scenario)

        predictions: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        next_hour_ready = \
                False if predictions[predictions[f"{scenario}_hour"] == to_hour].empty else True
                
        previous_hour_ready = \
                False if predictions[predictions[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            prediction_to_use = predictions[predictions[f"{scenario}_hour"] == to_hour]

        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            prediction_to_use = predictions[predictions[f"{scenario}_hour"] == from_hour]
            current_hour = from_hour

        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

    if not prediction_to_use.empty:
        st.sidebar.write("✅ Model's predictions received")
        tracker.next()

    return prediction_to_use


@st.cache_data
def make_geodataframe(scenario: str, geojson: dict) -> GeoDataFrame:
    
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

    return data
