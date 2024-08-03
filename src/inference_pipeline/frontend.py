import json
from typing import Any

import pydeck
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

from datetime import datetime, timedelta, UTC
from streamlit_option_menu import option_menu

from src.plot import plot_one_sample
from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


displayed_scenario_names = {"start": "Departures", "end": "Arrivals"}

st.title("Hourly Trip Predictor for Chicago's Divvy Bikes System")
current_hour = pd.to_datetime(datetime.now(UTC)).floor("H")
st.header(f"{current_hour} UTC")

pred_progress_bar = st.sidebar.header("⚙️ Working Progress")
pred_progress_bar = st.sidebar.progress(value=0)

n_steps = 7


def make_main_menu() -> str:

    with st.sidebar:
        return option_menu(
            menu_title="Main Menu",
            options=["Plots", "Predictions"],
            menu_icon="list_nested",
            icons=["bar-chart-fill", "bullseye"]
        )


def load_geodata(scenario: str, indexer: str = "two") -> pd.DataFrame | gpd.GeoDataFrame:
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
            with open(INDEXER_TWO/f"{scenario}_geodata_indexer_two.json") as file:
                geodata_dict = json.load(file)

            coordinates = [value[0] for value in geodata_dict.values()]
            station_ids = [value[1] for value in geodata_dict.values()]

            geodata = gpd.GeoDataFrame(
                {
                    f"{scenario}_station_names": geodata_dict.keys(),
                    f"{scenario}_station_ids": station_ids,
                    "coordinates": coordinates
                }
            )

        st.sidebar.write("✅ Station IDs & Coordinates Obtained...")
        return geodata


def get_hourly_predictions(
    scenario: str,
    model_name: str, 
    from_hour: datetime = current_hour - timedelta(days=1),
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
    with st.spinner(text="Fetching model predictions from the feature store..."):

        inferrer = InferenceModule(scenario=scenario)

        predictions_df: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        print(predictions_df.head())
        breakpoint()

        next_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == to_hour].empty else True

        if next_hour_ready: 
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == to_hour]

        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == from_hour]
            current_hour = from_hour

        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

        if not predictions_to_use.empty:
            st.sidebar.write("✅ Dataframe containing the model's predictions received...")
            pred_progress_bar.progress(2 / n_steps)
    
    return predictions_to_use


@st.cache_data
def provide_features(scenario: str, target_date: datetime) -> pd.DataFrame:
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
        features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=False)

        st.sidebar.write("✅ Fetched features for inference")
        pred_progress_bar.progress(1 / n_steps)
        return features 


def color_scaling_map_locations(
    value: int, 
    min_value: int,
    max_value: int, 
    start_colour: tuple, 
    stop_colour: tuple
    ) -> tuple[float | Any]:
    """
    Use linear interpolation to perform colour scaling on the predicted values. This provides us
    with a spectrum of colours for the prediction values.

    Credit to Pau Labarta Bajo and https://stackoverflow.com/a/10907855

    Args:
        value (int): _description_
        min_value (int): _description_
        max_value (int): _description_
        start_colour (tuple): _description_
        stop_colour (tuple): _description_

    Returns:
        tuple[float]: _description_
    """
    
    f = float(
        (value - min_value) / (max_value - min_value)
    )

    return tuple(
        f * (b - a) + a for (a, b) in zip(start_colour, stop_colour)
    )


def make_map(geodata: pd.DataFrame) -> None:
    """

    Args:
        geodata:

    Returns:
        None
    """
    # Selected a random coordinate to use as a start position
    start_position = pydeck.ViewState(
        latitude=41.872866,
        longitude=-87.63363,
        zoom=10,
        max_zoom=20,
        pitch=45,
        bearing=0
    )

    geojson = pydeck.Layer(type="GeoJsonLayer", data=geodata)
    tooltip = {"html": "<b>Zone:</b> [{StationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_trips}"}

    deck = pydeck.Deck(
        layers=[geojson],
        initial_view_state=start_position,
        tooltip=tooltip
    )

    st.pydeck_chart(pydeck_obj=deck)
    pred_progress_bar.progress(3 / n_steps)

def prep_data_for_plotting(scenario: str, predictions: pd.DataFrame, geodata: pd.DataFrame) -> None:
    """

    Args:
        scenario: "start" or "end"
        predictions: the loaded predictions
        geodata: the dataframe of station IDs and coordinates

    Returns:
        None.
    """
    with st.spinner(text="Preparing data..."):
        data = pd.merge(
            left=geodata,
            right=predictions,
            right_on=f"{scenario}_station_id",
            left_on=f"{scenario}_station_id",
            how="inner"
        )

    # Establish the max and min values as well as the start and stop colours for the colour scaling.
    black, green = (0, 0, 0), (0, 255, 0)
    data["colour_scaling"] = data[f"predicted_{scenario}s"]
    max_prediction, min_prediction = data["colour_scaling"].max(), data["colour_scaling"].min()

    #  Perform color scaling
    data["fill_colour"] = data["colour_scaling"].apply(
        func=lambda x: color_scaling(
            value=x, min_value=min_prediction, max_value=max_prediction,start_colour=black, stop_colour=green
        )
    )

    pred_progress_bar.progress(4 / n_steps)

def plot_time_series(scenario: str, features: pd.DataFrame, predictions: pd.DataFrame):

    with st.spinner(text="Plotting time series data..."):
        row_indices = np.argsort(predictions[f"predicted_{scenario}s"].values)[::-1]
        n_to_plot = 10

        for row in row_indices[:n_to_plot]:
            station_id = predictions[f"{scenario}_station_id"].iloc[row]
            prediction = predictions[f"predicted_{scenario}s"].iloc[row]

            st.metric(
                label=f"Predicted {displayed_scenario_names[scenario]}", 
                value=int(prediction)
            )

            fig = plot_one_sample(
                example_station=row,
                features=features,
                targets=predictions[f"predicted_{scenario}s"],
                display_title=False
            )

            st.plotly_chart(figure_or_data=fig, theme="streamlit", use_container_width=True, width=1000)
 

def construct_page(model_name: str):
    """


    Args:
        model_name (str): _description_
    """
    menu_options = make_main_menu()

    if menu_options == "Plots":
        pass  # Plotting logic will be provided at a later time
    elif menu_options == "Predictions":
        user_scenario_choice: list[str] = st.sidebar.multiselect(
            label="Do you want predictions for the number of arrivals at or the departures from each station?",
            options=["Arrivals", "Departures"],
            placeholder="Please select one of the two options."
        )

        for scenario in displayed_scenario_names.keys():
            if displayed_scenario_names[scenario] in user_scenario_choice:

                features = provide_features(scenario=scenario, target_date=current_hour)
                geo_df = load_geodata(scenario=scenario)

                predictions_df: pd.DataFrame = get_hourly_predictions(scenario=scenario, model_name=model_name)
                


if __name__ == "__main__":
    construct_page(model_name="lightgbm")
