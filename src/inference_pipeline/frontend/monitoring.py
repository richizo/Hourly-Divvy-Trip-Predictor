"""
This module provides code that delivers the data used for model monitoring, computes the mean 
absolute error of predictions against historical data, and illustrates the model error over 
time using bar chart(s).
"""
import pandas as pd 
import streamlit as st 
import plotly.express as px
from datetime import timedelta

from sklearn.metrics import mean_absolute_error

from src.setup.config import config
from src.monitoring import load_predictions_and_historical_trips
from src.inference_pipeline.frontend.tracker import ProgressTracker


@st.cache_data
def fetch_from_monitoring_feature_view(scenario: str, model_name: str = "xgboost") -> pd.DataFrame:
    """
    Fetch historical and prediction data.

    Args:
        scenario (str): "start" or "end".
        model_name (str, optional): Defaults to "xgboost".

    Returns:
        pd.DataFrame: the data to be used for model monitoring
    """
    with st.spinner("Getting both predicted and historical trip data from the feature store"):

        return load_predictions_and_historical_trips(
            scenario=scenario,
            model_name=model_name,
            from_date=config.current_hour - timedelta(days=40),
            to_date=config.current_hour
        )


def plot_bar_chart(data: pd.DataFrame, x_axis: str, y_axis: str) -> None:
    """
    Create a bar chart and display it in the streamlit interface

    Args:
        data (pd.DataFrame): the data used for monitoring
        x_axis (str): the name of the variable associated with the x-axis
        y_axis (str): the name of the variable associated with the y-axis
    """
    fig = px.bar(data_frame=data, x=x_axis, y=y_axis, template="plotly_dark")
    st.plotly_chart(figure_or_data=fig, theme="streamlit")


@st.cache_resource
def plot_error_per_hour(scenario: str, data_to_monitor: pd.DataFrame, aggregate_or_top: str) -> None:
    """
    Compute the mean absolute error (in aggregate terms, for the top 10 stations, or both) of the predicted
    values with respect to the historical ones, and plot it using a bar chart.

    Args:
        scenario (str): "start" or "end"
        data_to_monitor (pd.DataFrame): the data to base the monitoring procedure on
        aggregate_or_top (str): a string that determines whether we want the errors in aggregate terms or 
                                only for the top ten stations
    """
    if aggregate_or_top == "aggregate error":
        with st.spinner("Plotting aggregate mean absolute error (MAE) by the hour"):
            st.header(body="Mean Absolute Error (MAE) Per Hour")

            error_per_hour = (
                data_to_monitor
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

            top_locations = (
                data_to_monitor
                .groupby(f"{scenario}_station_id")["trips"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .head(10)[f"{scenario}_station_id"]
            )

            for station_id in top_locations:
                error_per_hour = (
                    monitoring_data[monitoring_data[f"{scenario}_station_id"] == station_id]
                    .groupby(f"{scenario}_hour")
                    .apply(lambda f: mean_absolute_error(y_true=f["trips"], y_pred=f[f"predicted_{scenario}s"]))
                    .reset_index()
                    .rename(columns={0: "mean_absolute_error"})
                    .sort_values(by=f"{scenario}_hour")
                )

                plot_bar_chart(data=error_per_hour, x_axis=f"{scenario}_hour", y_axis="mean_absolute_error")
                    

if __name__ != "__main__":

    user_scenario_choice = st.multiselect(
        label="Do you want to monitor the performance of the model used to predict arrivals or departures?",
        placeholder="Please select an option",
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

            user_plot_choice = st.multiselect(
                label="Do you want to monitor the model's error in aggregate, or for only the most active stations?",
                placeholder="Please select one of the three given options",
                options=["Aggregate Error", "Top Stations Only", "Both"]
            )

            plot_error_per_hour(scenario=scenario, data_to_monitor=monitoring_data, aggregate_or_top=user_plot_choice)
            tracker.next()

        else:
            continue
