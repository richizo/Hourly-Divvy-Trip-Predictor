import json
from typing import Any

import numpy as np

import streamlit as st

from tqdm import tqdm
from loguru import logger

import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime, timedelta, UTC

from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry

from src.setup.config import choose_displayed_scenario_name, config 
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


current_hour = pd.to_datetime(datetime.now(UTC)).floor("H")
        

class Loader:

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario.lower()
        assert scenario.lower() == "start" or "end"
        self.displayed_scenario_name = choose_displayed_scenario_name()[self.scenario]
        
    @st.cache_data
    def load_geojson(self, indexer: str = "two") -> dict:
        """

        Args:
            indexer (str, optional): _description_. Defaults to "two".

        Returns:
            dict: _description_
        """
        with st.spinner(text="Getting the coordinates of each station..."):

            if indexer == "one":

                with open(GEOGRAPHICAL_DATA / f"rounded_{self.scenario}_points_and_new_ids.geojson") as file:
                    points_and_ids = json.load(file)

                loaded_geodata = pd.DataFrame(
                    {
                        f"{self.scenario}_station_id": points_and_ids.keys(), 
                        "coordinates": points_and_ids.values()
                    }
                )

                reverse_geocoding = ReverseGeocoding(scenario=self.scenario, geodata=loaded_geodata)
                station_names_and_locations = reverse_geocoding.reverse_geocode()

                updated_geodata = reverse_geocoding.put_station_names_in_geodata(
                    station_names_and_coordinates=station_names_and_locations
                )

                return updated_geodata
            
            elif indexer == "two":
                with open(INDEXER_TWO/f"{self.scenario}_geodata.geojson") as file:
                    geodata_dict = json.load(file)

        st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")
        return geodata_dict


    @st.cache_data
    def get_features(self, target_date: datetime, geocode: bool = False) -> pd.DataFrame:
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
            inferrer = InferenceModule(scenario=self.scenario)
            features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=geocode)

        return features 


    @st.cache_data
    def get_hourly_predictions(
        self,
        model_name: str = "lightgbm", 
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
        with st.spinner(
            text=f"Fetching predicted {self.displayed_scenario_names.lower()} from the feature store..."
        ):
            inferrer = InferenceModule(scenario=self.scenario)

            predictions: pd.DataFrame = inferrer.load_predictions_from_store(
                model_name=model_name,
                from_hour=from_hour, 
                to_hour=to_hour
            )

            next_hour_ready = \
                 False if predictions[predictions[f"{self.scenario}_hour"] == to_hour].empty else True
                 
            previous_hour_ready = \
                 False if predictions[predictions[f"{self.scenario}_hour"] == from_hour].empty else True

            if next_hour_ready: 
                prediction_to_use = predictions[predictions[f"{self.scenario}_hour"] == to_hour]

            elif previous_hour_ready:
                st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
                prediction_to_use = predictions[predictions[f"{self.scenario}_hour"] == from_hour]
                current_hour = from_hour

            else:
                raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

        return prediction_to_use


    #@st.cache_data
    def make_geodataframe(self, geojson: dict) -> GeoDataFrame:
        
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
                f"{self.scenario}_station_name": station_names,
                f"{self.scenario}_station_id": station_ids,
                "latitudes": latitudes,
                "longitudes": longitudes
            }
        )

        #  data["coordinates"] = data["coordinates"].apply(lambda x: Point(x[0], x[1]))
        #  geodata = GeoDataFrame(data=data, geometry="coordinates")

        #  geodata.set_crs(epsg=4326, inplace=True)

        progress_bar.progress(1 / n_steps)
        return data


class ProgressTracker:

    def __init__(self, n_steps: int):
        
        self.current_step = 1
        self.n_steps = n_steps
        self.progress_bar = st.sidebar.header("⚙️ Working Progress")
        self.progress_bar = st.sidebar.progress(value=0)

    def next(self, stage: int) -> None:
        self.progress_bar.progress(self.current/self.n_steps)
        self.current_step += 1 


