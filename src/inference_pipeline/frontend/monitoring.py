import pandas as pd 
import streamlit as st 
import plotly.express as px

from datetime import datetime, timedelta, UTC
from sklearn.metrics import mean_absolute_error
from loguru import logger

from src.setup.config import config
from src.monitoring import load_predictions_and_historical_trips
from src.inference_pipeline.frontend.main import ProgressTracker


@st.cache_data
def fetch_from_monitoring_feature_view(scenario: str, model_name: str = "lightgbm") -> pd.DataFrame:
    """
    Fetch historical and predicteion data 

    Args:
        scenario (str): _description_
        model_name (str, optional): _description_. Defaults to "lightgbm".

    Returns:
        pd.DataFrame: _description_
    """
        
    with st.spinner("Getting both predicted and historical trip data from the feature store"):
        current_hour = pd.to_datetime(datetime.now(UTC)).floor(freq="H")

        monitoring_data = load_predictions_and_historical_trips(
            scenario=scenario,
            model_name=model_name,
            from_date=current_hour - timedelta(days=7),
            to_date=current_hour
        )

    return monitoring_data


def plot_bar_chart(data: pd.DataFrame, x_axis: str, y_axis: str):
    fig =  px.bar(data_frame=data, x=x_axis, y=y_axis, template="plotly_dark")
    st.plotly_chart(figure_or_data=fig, theme="streamlit")
    tracker.next()


@st.cache_resource
def plot_error_per_hour(scenario: str, data_to_monitor: pd.DataFrame, aggregate_or_top: str) -> None:

    if aggregate_or_top == "aggregate error":
        with st.spinner("Plotting aggregate mean absolute error (MAE) by the hour"):
            st.header(body="Mean Absolute Error (MAE) Per Hour")

            error_per_hour = (
                data
                .groupby(f"{scenario}_hour")
                .apply(lambda f: mean_absolute_error(y_true=f["trips"], y_pred=f[f"predicted_{scenario}s"]))
                .reset_index()
                .rename(columns={0: "mean_absolute_error"})
                .sort_values(by=f"{scenario}_hour")
            )

            plot_bar_chart(data=error_per_hour, x_axis=f"{scenario}_hour", y_axis="mean_absolute_error")

    elif aggregate_or_top == "top stations only":

        with st.spinner(text="Plotting aggregate mean absolute error (MAE) by the hour"):            
            st.header(body="Mean Absolute Error (MAE) Per Station Per Hour")

            error_per_hour = top_locations(
                data_to_monitor
                .groupby(f"{scenario}_station_id")["trips"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .head(10)[f"{scenario}_station_id"]
            )

            for station_id in top_locations:
                plot_bar_chart(data=error_per_hour, x_axis=f"{scenario}_hour", y_axis="mean_absolute_error")
                    

if __name__ != "__main__":

    user_scenario_choice = st.sidebar.multiselect(
        label="Do you want to monitor the performance of the model used to predict arrivals or departures?",
        placeholder="Please select one of the two given options",
        options=config.displayed_scenario_names.values()
    )

    for scenario in config.displayed_scenario_names.keys():

        if config.displayed_scenario_names[scenario] in user_scenario_choice:
            tracker = ProgressTracker(n_steps=4)
            tracker.next()

            monitoring_data = fetch_from_monitoring_feature_view(scenario=scenario)
                
            if not monitoring_data.empty:
                st.sidebar.write("✅ Model Predictions and Historical Data have been fetched")
                tracker.next()
            else:
                st.sidebar.write("Dataframe empty")

            user_plot_choice = st.sidebar.multiselect(
                label="Do you want to monitor the model's error in aggregate, or for only the most active stations?",
                placeholder="Please select one of the three given options",
                options=["Aggregate Error", "Top Stations Only" , "Both"]
            )

            plot_error_per_hour(scenario=scenario, data_to_monitor=monitoring_data, aggregate_or_top=user_plot_choice)
            tracker.next()

        else:
            break