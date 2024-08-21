import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from src.setup.config import choose_displayed_scenario_name, config
from src.inference_pipeline.inference import InferenceModule

from src.inference_pipeline.frontend.data import (
    load_geojson, load_geodata, get_features, get_ids_and_names, prepare_geodata_df
)


class ProgressTracker:

    def __init__(self, n_steps: int):
        
        self.current_step = 0
        self.n_steps = n_steps
        self.progress_bar = st.sidebar.header("⚙️ Working Progress")
        self.progress_bar = st.sidebar.progress(value=0)

    def next(self) -> None:
        self.current_step += 1 
        self.progress_bar.progress(self.current_step/self.n_steps)


displayed_scenario_names = choose_displayed_scenario_name()
tracker = ProgressTracker(n_steps=4)


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
    with st.spinner(text=f"Fetching predicted {displayed_scenario_names[scenario].lower()} from the feature store..."):
        inferrer = InferenceModule(scenario=scenario)

        predictions: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        next_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            prediction_to_use = predictions[predictions[f"{scenario}_hour"] == to_hour]
        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            prediction_to_use = predictions[predictions[f"{scenario}_hour"] == from_hour]
        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

    if not prediction_to_use.empty:
        st.sidebar.write("✅ Model's predictions received")
        tracker.next()

    return prediction_to_use
    

def get_prediction_per_station(scenario: str, predictions_df: pd.DataFrame) -> dict[str, float]:

    station_ids = predictions_df[f"{scenario}_station_id"].values
    predictions = predictions_df[f"predicted_{scenario}s"].values
    geodata: dict = load_geodata(scenario=scenario)
    ids_and_names = get_ids_and_names(geodata=geodata)

    ids_and_predictions: dict[int, float] = {
        code: prediction for code, prediction in zip(station_ids, predictions) if prediction is not None
    }

    return {
        ids_and_names[station_id]: ids_and_predictions[station_id] for station_id in ids_and_predictions.keys()
    }
           

def deliver_predictions():

    user_scenario_choice: list[str] = st.sidebar.multiselect(
        label="Are you looking to view the number of predicted arrivals or departures?",
        options=["Arrivals", "Departures"],
        placeholder="Please select one of the two options."
    )

    for scenario in displayed_scenario_names.keys():
        if displayed_scenario_names[scenario] in user_scenario_choice:

            # Prepare geodata   
            geojson = load_geojson(scenario=scenario)
            geodata = prepare_geodata_df(scenario=scenario, geojson=geojson)
            tracker.next()
            
            # Fetch features and predictions<
            features = get_features(scenario=scenario, target_date=config.current_hour)
            predictions = get_hourly_predictions(scenario=scenario)

            choose_station = st.selectbox(
                label=f"Which station would you like predicted {displayed_scenario_names[scenario]}s for?"
            
            )
        