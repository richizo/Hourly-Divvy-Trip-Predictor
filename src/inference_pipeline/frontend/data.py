import os 
import json 
import pandas as pd
import streamlit as st 

from loguru import logger 

from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule


def rerun_feature_pipeline(exception_to_check: FileNotFoundError):
    """

    Args:
        exception_to_check (FileNotFoundError): _description_
    """
    
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)

            # Check whether     
            except exception_to_check as error:
                logger.error(error)
                logger.error(
                    "The JSON file which contains station details is missing. The feature pipeline is now being re-run..."
                )

                # Run the full preprocessing script, even though we only need the indexing to be done.
                processor = DataProcessor(year=config.year, for_inference=False)
                processor.make_training_data(geocode=False)
                
                # Rerun the wrapped function
                return fn(*args, **kwargs)
                
        return decorator


@st.cache_data
@rerun_feature_pipeline(exception_to_check=FileNotFoundError)
def load_geodata(scenario: str) -> dict:

    if len(os.listdir(INDEXER_ONE)) != 0:
        geodata_path = INDEXER_ONE / f"{scenario}_geodata.json"
    elif len(os.listdir(INDEXER_TWO)) != 0:
        geodata_path = INDEXER_TWO / f"{scenario}_geodata.json"
    else:
        raise FileNotFoundError("No geodata has been made. Running the feature pipeline...")

    with open(geodata_path) as file:
        geodata = json.load(file)

    return geodata 


@st.cache_data
def get_ids_and_names(geodata: dict) -> dict[str, int]:
    ids_and_names = [(station_details["station_id"], station_details["station_name"]) for station_details in geodata]
    return {station_id: station_name for station_ids, station_name in ids_and_names}    


@st.cache_data
@rerun_feature_pipeline(exception_to_check=FileNotFoundError)
def load_geojson(scenario: str) -> dict:
    """

    Args:
        indexer (str, optional): _description_. Defaults to "two".

    Returns:
        dict: _description_
    """
    with st.spinner(text="Getting the coordinates of each station..."):
        
        if len(os.listdir(INDEXER_ONE)) != 0:

            with open(INDEXER_ONE / f"rounded_{scenario}_points_and_new_ids.geojson") as file:
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
            with open(INDEXER_TWO/f"{scenario}_geojson.geojson") as file:
                geodata_dict = json.load(file)      
                
        else:
            raise FileNotFoundError("No geojson to used for plotting has been made. Running the feature pipeline...")


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
