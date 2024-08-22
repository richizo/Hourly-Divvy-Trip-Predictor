import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from loguru import logger
from src.inference_pipeline.inference import InferenceModule
from src.setup.config import config

from src.inference_pipeline.frontend.data import (
    load_geojson, load_geodata, get_features, get_ids_and_names, prepare_geodata_df
)

from src.inference_pipeline.frontend.main import ProgressTracker


@st.cache_data
def get_all_predictions(
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
    with st.spinner(text=f"Fetching predicted {config.displayed_scenario_names[scenario].lower()} from feature store"):
        inferrer = InferenceModule(scenario=scenario)

        predictions: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        next_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            fetched_predictions = predictions[predictions[f"{scenario}_hour"] == to_hour]
        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            fetched_predictions = predictions[predictions[f"{scenario}_hour"] == from_hour]
        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

    return fetched_predictions
    

@st.cache_data
def get_prediction_per_station(scenario: str, predictions_df: pd.DataFrame) -> dict[str, float]:
    """
    Go through the dataframe of predictions and obtain the prediction associated with each station
    ID. Then get the name of each station, and return a dictionary with names as keys and predictions 
    as values.

    Args:
        scenario (str): "start" or "end"
        predictions_df (pd.DataFrame): the dataframe of predictions downloaded from the feature store.

    Returns:
        dict[str, float]: _description_
    """

    station_ids = predictions_df[f"{scenario}_station_id"].values
    predictions = predictions_df[f"predicted_{scenario}s"].values

    geodata: dict = load_geodata(scenario=scenario)
    ids_and_names = get_ids_and_names(geodata=geodata)

    ids_and_predictions: dict[int, float] = {
        code: prediction for code, prediction in zip(station_ids, predictions) if prediction is not None
    }

    if len(predictions_df[f"{scenario}_station_id"].unique()) == len(ids_and_predictions.keys()):
        logger.success("Predictions retrieved for all stations")

    return {
        ids_and_names[station_id]: ids_and_predictions[station_id] for station_id in ids_and_predictions.keys()
    }
           

if __name__ != "__main__":  
    tracker = ProgressTracker(n_steps=2)
    
    user_scenario_choice: list[str] = st.sidebar.multiselect(
        label="Are you looking to view the number of predicted arrivals or departures?",
        placeholder="Please select one of the two options",
        options=["Arrivals", "Departures"]
    )

    tracker.next()
    for scenario in config.displayed_scenario_names.keys():
        arrival_or_departure = config.displayed_scenario_names[scenario]
        if arrival_or_departure in user_scenario_choice:

            predictions_df = get_all_predictions(scenario=scenario)

            if not predictions_df.empty:
                st.sidebar.write("✅ All models predictions received")
                tracker.next()

            predictions_per_station = get_prediction_per_station(scenario=scenario, predictions_df=predictions_df)

            chosen_station = st.selectbox(
                label=f"Which station would you like predicted {arrival_or_departure} for?",
                options=list(predictions_per_station.keys()),
                placeholder="Please choose a station"
            )
 
            requested_prediction = int(predictions_per_station[chosen_station])

            st.write(
                f"We predict {requested_prediction} {config.displayed_scenario_names[scenario].lower()} here \
                    in the next hour"
            )
