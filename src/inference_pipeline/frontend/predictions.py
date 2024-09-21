"""
Contains code responsible for fetching predictions from the feature store 
and displaying it in the streamlit interface. 
"""
import pandas as pd
from tqdm import tqdm
import streamlit as st
from loguru import logger
from datetime import datetime, timedelta
from streamlit_extras.colored_header import colored_header

from src.setup.config import config
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.frontend.main import ProgressTracker
from src.inference_pipeline.frontend.data import load_raw_local_geodata, get_ids_and_names


@st.cache_data
def get_all_predictions(
        model_name="xgboost",
        from_hour=config.current_hour - timedelta(hours=1),
        to_hour=config.current_hour
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download all the predictions for all the stations from one hour to another

    Args:
        model_name: the name of the model which was trained to produce the predictions we now seek.
        from_hour (datetime, optional): From which hour we want to fetch predictions. Defaults to the previous hour.
        to_hour (datetime, optional): the hour we want predictions for. Defaults to the current hour.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: a list of dataframes of predictions for both arrivals and departures
    """
    prediction_dataframes = []
    for scenario in config.displayed_scenario_names.keys():

        infer = InferenceModule(scenario=scenario)
        predictions: pd.DataFrame = infer.load_predictions_from_store(
            model_name=model_name, 
            from_hour=from_hour, 
            to_hour=to_hour
        )

        geodata = load_raw_local_geodata(scenario=scenario)
        ids_and_names = get_ids_and_names(local_geodata=geodata)
        predictions["station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)
        prediction_dataframes.append(predictions)

    predicted_departures, predicted_arrivals = prediction_dataframes[0], prediction_dataframes[1] 
    
    breakpoint()
    return predicted_departures, predicted_arrivals


@st.cache_data
def extract_predictions_for_this_hour(
        predicted_starts: pd.DataFrame,
        predicted_ends: pd.DataFrame,
        from_hour: datetime,
        to_hour: datetime,
        include_station_names: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Initialise an inference object, and load a dataframe of predictions from its dedicated feature group
    on the offline feature store. We then fetch the most recent prediction if it is available, or the second
    most recent (the one from an hour before)

    Args:
        predicted_starts (pd.DataFrame): the dataframe of of all predicted departures for all stations and hours.
        predicted_ends (pd.DataFrame): the dataframe of of all predicted arrivals for all stations and hours.
        include_station_names (bool): whether to add a column of station names to the predictions. Defaults to True.
        from_hour (datetime, optional): From which hour we want to fetch predictions. Defaults to the previous hour.
        to_hour (datetime, optional): the hour we want predictions for. Defaults to the current hour.

    Raises:
        Exception: In the event that the predictions for the current hour, or the previous one cannot be obtained.
                    This exception suggests that the feature pipeline may not be working properly.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: dataframes containing predicted arrivals and departures for this, or
                                           the previous hour.
    """
    extracted_predictions = []
    scenario_and_predictions = {"start": predicted_starts, "end": predicted_ends}

    for scenario in scenario_and_predictions.keys():

        predictions = scenario_and_predictions[scenario]
        next_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready:
            predictions_for_target_hour = predictions[predictions[f"{scenario}_hour"] == to_hour]
        elif previous_hour_ready:
            st.write("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            predictions_for_target_hour = predictions[predictions[f"{scenario}_hour"] == from_hour]
        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

        if include_station_names:
            target_hour = from_hour if next_hour_ready else to_hour
            logger.info(f"Working to attach the station names to the predictions for {target_hour}")
            raw_local_geodata = load_raw_local_geodata(scenario=scenario)
            ids_and_names = get_ids_and_names(geodata=raw_local_geodata)

            new_column_of_names = []
            station_ids = predictions_for_target_hour[f"{scenario}_station_id"].values
            for station_id in tqdm(
                    iterable=station_ids,
                    desc="Grabbing the names of all the stations we have predictions for"
            ):
                new_column_of_names.append(ids_and_names[station_id])

            predictions_for_target_hour = pd.concat(
                [predictions_for_target_hour, pd.Series(new_column_of_names)], axis=0
            )

        extracted_predictions.append(predictions_for_target_hour)

    return extracted_predictions[0], extracted_predictions[1]


@st.cache_data
def get_predictions_per_station(scenario: str, predictions: pd.DataFrame) -> dict[str, float]:
    """
    Go through the dataframe of predictions and obtain the prediction associated with each station
    ID. Then get the name of each station, and return a dictionary with names as keys and predictions
    as values.
z
    Args:
        scenario (str): "start" or "end"
        predictions (pd.DataFrame): the dataframe of predictions downloaded from the feature store.

    Returns:
        dict[str, float]:
    """
    station_ids = predictions[f"{scenario}_station_id"].values
    predictions = predictions[f"predicted_{scenario}s"].values

    logger.info(f"Predictions for {len(station_ids)} stations were fetched")
    geodata: list[dict] = load_raw_local_geodata(scenario=scenario)
    ids_and_names = get_ids_and_names(local_geodata=geodata)

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

        breakpoint()

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
    get_all_predictions()