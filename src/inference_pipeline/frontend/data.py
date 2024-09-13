"""
This module contains code responsible for loading the various pieces of data 
that will be used to deliver the predictions to the streamlit interface.
"""
import os
import json 
import pandas as pd
import streamlit as st 

from loguru import logger
from datetime import datetime, UTC

from src.setup.config import config
from src.setup.paths import INDEXER_ONE, INDEXER_TWO

from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule


def rerun_feature_pipeline():
    """
    This will be a decorator that will be applied to a few functions that will benefit from its functionality.
    It provides logic that allows the wrapped function to be run if a certain exception is not raised, and
    run the full feature pipeline if the exception is raised. Generally, the functions that will use this will
    depend on the loading of some file that was generated during the preprocessing phase of the feature pipeline.
    Running the feature pipeline will allow for the file in question to be generated if isn't present, and then
    run the wrapped function afterwards.
    """
    def decorator(fn: callable):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except FileNotFoundError as error:
                logger.error(error)
                message = "The JSON file containing station details is missing. Running feature pipeline again..."
                logger.warning(message)
                st.spinner(message)

                processor = DataProcessor(year=config.year, for_inference=False)
                processor.make_training_data(geocode=False)
                return fn(*args, **kwargs)
        return wrapper
    return decorator


@st.cache_data
@rerun_feature_pipeline()
def load_geodata(scenario: str) -> dict:
    """
    Load the json file that contains the geographical information for 
    each station.

    Args:
        scenario (str): "start" or "end" 

    Raises:
        FileNotFoundError: raised when said json file cannot be found. In that case, 
        the feature pipeline will be re-run. As part of this, the file will be created,
        and the function will then load the generated data.

    Returns:
        dict: the loaded json file as a dictionary
    """
    if len(os.listdir(INDEXER_ONE)) != 0:
        geodata_path = INDEXER_ONE / f"{scenario}_geodata.json"
    elif len(os.listdir(INDEXER_TWO)) != 0:
        geodata_path = INDEXER_TWO / f"{scenario}_geodata.json"
    else:
        raise FileNotFoundError("No geodata has been made. Running the feature pipeline...")

    with open(geodata_path, mode="r") as file:
        geodata = json.load(file)
    return geodata 


@st.cache_data
def get_ids_and_names(geodata: dict) -> dict[int, str]:
    """
    Extract the station IDs and names from the dictionary of station details.

    Args:
        geodata (dict): disctionary containing geographical details of each station

    Returns:
        dict[int, str]: station IDs as keys and station names as values
    """
    with st.spinner(text="Accumulating station details..."):
        ids_and_names = [(station_details["station_id"], station_details["station_name"]) for station_details in geodata]
        return {station_id: station_name for station_id, station_name in ids_and_names}


@rerun_feature_pipeline()
def load_geojson(scenario: str) -> dict:
    """
    Load the geojson file that was generated during the feature pipeline. It will be used to 
    generate the points on the map.

    Args:
        scenario (str): "start" or "end"

    Raises:
        FileNotFoundError: raised when said json file cannot be found. In that case, 
        the feature pipeline will be re-run. As part of this, the file will be created,
        and the function will then load the generated data.

    Returns:
        dict: the loaded geojson file.
    """
    with st.spinner(text="Getting the coordinates of each station..."):
        if len(os.listdir(INDEXER_ONE)) != 0:
            with open(INDEXER_ONE / f"rounded_{scenario}_points_and_new_ids.geojson", mode="r") as file:
                points_and_ids = json.load(file)

            loaded_geodata = pd.DataFrame(
                {
                    f"{scenario}_station_id": points_and_ids.keys(), 
                    "coordinates": points_and_ids.values()
                }
            )

            reverse_geocoding = ReverseGeocoding(scenario=scenario, geodata=loaded_geodata)
            station_names_and_locations = reverse_geocoding.reverse_geocode()

            geodata_dict = reverse_geocoding.put_station_names_in_geodata(
                station_names_and_coordinates=station_names_and_locations
            )
        
        elif len(os.listdir(INDEXER_TWO)) != 0:
            with open(INDEXER_TWO/f"{scenario}_geojson.geojson", mode="r") as file:
                geodata_dict = json.load(file)
        else:
            raise FileNotFoundError("No geojson to used for plotting has been made. Running the feature pipeline...")

    st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")
    return geodata_dict


@st.cache_data
def prepare_geodata_df(scenario: str, geojson: dict) -> pd.DataFrame:
    """
    Make a dataframe of geographical information out of the geojson file.

    Args:
        scenario (str): "start" or "end"
        geojson (dict): the geojson file containing stations and their geographical details.

    Returns:
        pd.DataFrame: the created dataframe.
    """
    with st.spinner(text="Preparing a dataframe of station details for plotting..."):

        coordinates = []
        station_ids = []
        station_names = []

        for detail_index in range(len(geojson["features"])):
            detail: dict = geojson["features"][detail_index]
            coordinates.append(detail["geometry"]["coordinate"])
            station_ids.append(detail["properties"]["station_id"])
            station_names.append(detail["properties"]["station_name"])

        latitudes = [point[1] for point in coordinates]
        longitudes = [point[0] for point in coordinates]

        geodata_df = pd.DataFrame(
            data={
                f"{scenario}_station_name": station_names,
                f"{scenario}_station_id": station_ids,
                "latitudes": latitudes,
                "longitudes": longitudes
            }
        )

        logger.info(f"There are {geodata_df[f"{scenario}_station_id"].unique()} stations in the {scenario} geodata")

    return geodata_df


@st.cache_data
def get_features(scenario: str, target_date: datetime, geocode: bool = False) -> pd.DataFrame:
    """
    Initiate an inference object and use it to get features until the target date.
    features that we will use to fuel the model and produce predictions.

    Args:
        scenario (str): _description_
        target_date (datetime): _description_
        geocode (bool):

    Returns:
        pd.DataFrame: the created (or fetched) features
    """
    with st.spinner(text="Getting a batch of features from the store..."):
        inferrer = InferenceModule(scenario=scenario)
        features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=geocode)

    st.sidebar.write("✅ Fetched features for inference")
    return features
