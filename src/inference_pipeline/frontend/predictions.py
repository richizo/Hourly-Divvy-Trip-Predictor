"""
Contains code responsible for fetching predictions from the feature store 
and displaying it in the streamlit interface. 
"""
import pandas as pd
import streamlit as st
from loguru import logger
from datetime import datetime, timedelta
from streamlit_extras.customize_running import center_running
from streamlit_extras.colored_header import colored_header

from src.setup.config import config
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.frontend.main import ProgressTracker
from src.inference_pipeline.frontend.data import load_geodata, get_ids_and_names


@st.cache_data
def get_all_predictions(
    scenario: str,
    model_name: str = "xgboost", 
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
            fetched_predictions = predictions[predictions[f"{scenario}_hour"] == from_
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
            st.write("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            fetched_predictions = predictions[predictions[f"{scenario}_hour"] == from_hour]
        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

    return fetched_predictions
    

@st.cache_data
def get_predictions_per_station(scenario: str, predictions_df: pd.DataFrame) -> dict[str, float]:
    """
    Go through the dataframe of predictions and obtain the prediction associated with each station
    ID. Then get the name of each station, and return a dictionary with names as keys and predictions 
    as values.

    Args:
        scenario (str): "start" or "end"
        predictions (pd.DataFrame): the dataframe of predictions downloaded from the feature store.

    Returns:
        dict[str, float]: 
    """
    station_ids = predictions_df[f"{scenario}_station_id"].values
    predictions = predictions_df[f"predicted_{scenario}s"].values

    logger.info(f"Predictions for {len(station_ids)} stations were fetched")

    geodata: dict = load_geodata(scenario=scenario)
    ids_and_names = get_ids_and_names(geodata=geodata)

    ids_and_predictions: dict[int, float] = {
        code: prediction for code, prediction in zip(station_ids, predictions) if prediction is not None
    }

    return {
        ids_and_names[station_id]: ids_and_predictions[station_id] for station_id in ids_and_predictions.keys()
    }


def deliver_predictions(options_and_colours: dict, user_choice: str):

    with st.spinner(
        f"Loading predicted {options_and_colours[user_choice]}[{user_choice.lower()}] for various stations"
    ):
        options_and_scenarios = {
            option: scenario for scenario, option in config.displayed_scenario_names.items()
        }

        scenario = options_and_scenarios[user_choice]
        tracker = ProgressTracker(n_steps=2)
        predictions_df = get_all_predictions(scenario=scenario)

        if not predictions_df.empty:
            st.sidebar.write(f"✅ Predicted {user_choice} received")
            tracker.next()
        
        predictions_per_station = get_predictions_per_station(scenario=scenario, predictions_df=predictions_df)  
        
        chosen_station = st.selectbox(
            label=f"Which :blue[station's] predicted :red[{user_choice.lower()}] would you like to view?",
            options=list(predictions_per_station.keys()),
            placeholder="Please choose a station"
        )

        tracker.next()
        st.sidebar.write("✅ Results presented")
        requested_prediction = int(predictions_per_station[chosen_station])

        st.write(
            f"{options_and_colours[user_choice]}[{requested_prediction} {user_choice.lower()}] at \
            :blue[{chosen_station}] in the next hour"
        )


if __name__ != "__main__":    

    try:
        options_and_colours = {"Arrivals": ":green", "Departures": ":orange"}

        user_choice = st.selectbox(
            options=["Select an option", "Arrivals", "Departures"],
            label="Please specify whether you'd like to see the predicted :green[arrivals] or :orange[departures]. \
                If you'd like to see both, please select one, wait for the results, and then select the other."
        )

        for scenario in config.displayed_scenario_names.keys():
            arrival_or_departure = config.displayed_scenario_names[scenario]
            if arrival_or_departure in user_choice:
                deliver_predictions(options_and_colours=options_and_colours, user_choice=arrival_or_departure)
    
    except Exception as error:
        logger.error(error)
