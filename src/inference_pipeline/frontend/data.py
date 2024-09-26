"""
This module contains code responsible for loading the various pieces of data 
that will be used to deliver the predictions to the streamlit interface.
"""
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

from tqdm import tqdm
from pathlib import Path
from shapely import Point
from loguru import logger
from zipfile import ZipFile
from datetime import datetime
from geopandas import GeoDataFrame

from src.setup.config import config
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER, GEOGRAPHICAL_DATA

from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.backend.inference import InferenceModule
from src.feature_pipeline.feature_engineering import ReverseGeocoder


def make_geodataframes() -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Create dataframes containing the geographical details of each station using both
    arrival and departure data, and return them

    Returns:
        tuple[GeoDataFrame, GeoDataFrame]: geodataframes for arrivals and departures
    """
    geo_dataframes = []
    for scenario in config.displayed_scenario_names.keys():
        points = []
        station_names = []
        station_details: list[dict] = load_raw_local_geodata(scenario=scenario)

        for detail in tqdm(iterable=station_details, desc="Collecting station details"):
            points.append(detail["coordinates"])
            station_names.append(detail["station_name"])

        raw_geodata = GeoDataFrame(
            geometry=[Point(coordinate) for coordinate in points],
            data={
                "station_name": station_names,
                "coordinates": points
            }
        )

        raw_geodata = raw_geodata.set_crs(epsg=4326)
        geo_dataframes.append(raw_geodata)

    start_geodataframe, end_geodataframe = geo_dataframes[0], geo_dataframes[1]
    return start_geodataframe, end_geodataframe


def reconcile_geodata(start_geodataframe: GeoDataFrame, end_geodataframe: GeoDataFrame) -> GeoDataFrame:
    """
    To avoid redundancy, and provide a consistent experience, we will render a single map. Consequently, I can 
    only use stations that are common to both arrival and departure datasets. This function finds the stations 
    common to the

    Returns:

    """
    larger_dataframe = start_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else end_geodataframe
    smaller_dataframe = end_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else start_geodataframe

    shared_stations_bool = np.isin(element=larger_dataframe["station_name"], test_elements=smaller_dataframe["station_name"])
    common_data = larger_dataframe.loc[shared_stations_bool, :]

    logger.warning(
        f"{len(larger_dataframe) - len(common_data)} stations were discarded because they were not common to both datasets"
    )

    return common_data


class ExternalShapeFile:
    """
    Allows us to download shapefiles and load them for later processing.
    """
    def __init__(self, map_type: str):
        """
        Args:
            map_type: a string that specifies the type of map whose shapefile we would like.
        """
        urls = {
            "pedestrian": "https://data.cityofchicago.org/api/geospatial/v6kn-gc9b?fourfour=v6kn-gc9b&cacheBust=\
                            1712775948&date=20240920&accessType=DOWNLOAD&method=export&format=Shapefile",

            "divvy_stations": "https://data.cityofchicago.org/api/geospatial/bbyy-e7gq?fourfour=bbyy-e7gq&cacheBust=\
                              1726743616&date=20240920&accessType=DOWNLOAD&method=export&format=Shapefile"
        }

        self.file_names = {
            "pedestrian": "pedestrian_streets_shapefile.zip",
            "divvy_stations": "divvy_stations_shapefile.zip"
        }

        self.map_type = map_type
        self.url = urls[map_type]
        self.zipfile_path = GEOGRAPHICAL_DATA / self.file_names[self.map_type]

    def download_archive(self) -> None:
        """
        Download the zipfile that contains the shapefile for the given map type.

        Returns:
            None
        """
        logger.info(f"Downloading shapefile for {self.map_type}s")
        response = requests.get(url=self.url)
        if response.status_code == 200:
            open(self.zipfile_path, mode="wb").write(response.content)
        else:
            raise Exception(f"The URL for {self.map_type}s is not available")

    def load_data_from_shapefile(self) -> GeoDataFrame:
        """
        Extract the contents of the downloaded archive to access the shapefile within, and then deliver it as a
        geo-dataframe.

        Returns:
            GeoDataFrame: the contents of the shapefile, rendered as a geo-dataframe.
        """
        if Path(self.zipfile_path).is_file():
            logger.success(f"The shapefile for {self.map_type}s is already saved to disk.")
        else:
            self.download_archive()

        with ZipFile(self.zipfile_path, mode="r") as zipfile:
            zipfile.extractall(GEOGRAPHICAL_DATA/self.file_names[:-4])

        return gpd.read_file(filename=self.zipfile_path / f"{self.file_names}[:-4].shp").to_crs("epsg:4326")


def rerun_feature_pipeline():
    """
    This is a decorator that provides logic which allows the wrapped function to be run if a certain exception 
    is not raised, and the full feature pipeline if the exception is raised. Generally, the functions that will 
    use this will depend on the loading of some file that was generated during the preprocessing phase of the 
    feature pipeline. Running the feature pipeline will allow for the file in question to be generated if isn't 
    present, and then run the wrapped function afterwards.
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
def load_raw_local_geodata(scenario: str) -> list[dict]:
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
        list[dict]: the loaded json file as a dictionary
    """
    if len(os.listdir(ROUNDING_INDEXER)) != 0:
        geodata_path = ROUNDING_INDEXER / f"{scenario}_geodata.json"
    elif len(os.listdir(MIXED_INDEXER)) != 0:
        geodata_path = MIXED_INDEXER / f"{scenario}_geodata.json"
    else:
        raise FileNotFoundError("No geographical data has been made. Running the feature pipeline...")

    with open(geodata_path, mode="r") as file:
        raw_geodata = json.load(file)
        
    return raw_geodata 


@st.cache_data
def get_ids_and_names(local_geodata: list[dict]) -> dict[int, str]:
    """
    Extract the station IDs and names from the dictionary of station details obtained by loading the 
    local geodata.

    Args:
        local_geodata (list[dict]): list of dictionaries containing the geographical details of each station

    Returns:
        dict[int, str]: station IDs as keys and station names as values
    """
    with st.spinner(text="Accumulating station details..."):

        ids_and_names = [
            (station_details["station_id"], station_details["station_name"]) for station_details in local_geodata
        ]
        
        return {station_id: station_name for station_id, station_name in ids_and_names}


@rerun_feature_pipeline()
def load_local_geojson(scenario: str) -> dict:
    """
    Load the geojson file that was generated during the feature pipeline. It will be used to 
    generate the points on the map.

    Args:
        scenario (str): "start" or "end"

    Raises:
        FileNotFoundError: raised when said json file cannot be found. In that case, the feature pipeline
                           will be re-run. As part of this, the file will be created, and the function will
                           then load the generated data.
    Returns:
        dict: the loaded geojson file.
    """
    with st.spinner(text="Getting the coordinates of each station..."):
        if len(os.listdir(ROUNDING_INDEXER)) != 0:
            with open(ROUNDING_INDEXER / f"rounded_{scenario}_points_and_new_ids.geojson", mode="r") as file:
                points_and_ids = json.load(file)

            loaded_geodata = pd.DataFrame(
                {
                    f"{scenario}_station_id": points_and_ids.keys(), 
                    "coordinates": points_and_ids.values()
                }
            )

            reverse_geocoding = ReverseGeocoder(scenario=scenario, raw_geodata=loaded_geodata)
            station_names_and_locations = reverse_geocoding.reverse_geocode()

            geodata_dict = reverse_geocoding.put_station_names_in_geodata(
                station_names_and_coordinates=station_names_and_locations
            )
        
        elif len(os.listdir(MIXED_INDEXER)) != 0:
            with open(MIXED_INDEXER / f"{scenario}_geojson.geojson", mode="r") as file:
                geodata_dict = json.load(file)
        else:
            raise FileNotFoundError("No geojson to used for plotting has been made. Running the feature pipeline...")

    st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")
    return geodata_dict


@st.cache_data
def prepare_df_of_local_geodata(scenario: str, geojson: dict) -> pd.DataFrame:
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

        logger.info(f"There are {geodata_df[f"{scenario}_station_id"].unique()} stations in the {scenario} raw_geodata")

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
