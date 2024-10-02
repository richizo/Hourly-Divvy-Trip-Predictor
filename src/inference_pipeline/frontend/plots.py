"""
Plot time series data for a given number of top stations.
"""

import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.express as px 

from tqdm import tqdm 
from plotly.graph_objects import Figure
from datetime import datetime, timedelta, UTC
from streamlit_extras.colored_header import colored_header

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, INFERENCE_DATA, GEOGRAPHICAL_DATA
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names
from src.inference_pipeline.backend.inference import InferenceModule


def get_station_name(station_id: int) -> str:
    """
    Look up the name of a station given that ID that it carries in the dataset in question.

    Args:
        station_id (int): the ID of the station of interest.

    Returns:
        str: the name of the station
    """
    with open(MIXED_INDEXER/"end_geodata.json", mode="r") as file:
        geodata = json.load(file)

    for station in tqdm(iterable=geodata, desc="Looking up the name of the station"):
        if station["station_id"] == station_id:
            return station["station_name"]


def plot_for_one_station(
    station_name: int,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    display_title: bool | None = True,
    targets: pd.Series | None = None
) -> Figure:
    """
    Plot the time series data for the given station.

    Args:
        station_name (int): the name of the station whose data we want to plot 
        features (pd.DataFrame): _description_
        predictions (pd.DataFrame): _description_
        display_title (str, optional): _description_. Defaults to None.
        targets (pd.Series | None, optional): _description_. Defaults to None.
    """
    station_features: pd.DataFrame = features[features[f"{scenario}_station_name"] == station_name]
    station_targets = targets[targets["station_name"] == station_name] if targets is not None else None

    station_features.to_parquet(INFERENCE_DATA/f"{scenario}_station_features.parquet")
    station_targets.to_parquet(INFERENCE_DATA/f"{scenario}_station_targets.parquet")
    breakpoint()

    columns_of_past_trips = [column for column in station_features.columns if column.startswith("trips_previous_")]
    cumulative_trips = [station_features[column] for column in columns_of_past_trips] + [station_targets]

    all_trip_dates = pd.date_range(
        start=station_features[f"{scenario}_hour"] - timedelta(hours=len(columns_of_past_trips)),
        end=station_features[f"{scenario}_hour"],
        freq="H" 
    )

    # Surely the whole hour column can't be here
    title = f"{config.displayed_scenario_names[scenario]} hour = {station_features[f"{scenario}_hour"]}, Station name =\
        {station_name}" if display_title else None

    figure = px.line(
        x=all_trip_dates,
        y=cumulative_trips,
        template="plotly-dark",
        markers=True,
        title=title
    )

    if targets is not None:

        figure.add_scatter(
            x=all_trip_dates[-1:], y=[station_targets], line_color="green", mode="markers", name="Actual value"
        )

    if predictions is not None:
        station_prediction = predictions[predictions[f"{scenario}_station_name"] ==  station_name]

        figure.add_scatter(
            x=all_trip_dates[-1:], 
            y=[station_prediction], 
            line_color="red",
            marker_symbol="x", 
            mode="markers", 
            name="Prediction"
        )

    return figure


if __name__ != "__main__":

    colored_header(
        label="Plots of Trips Over Time", 
        description="View plots of time series data of trips from the top 10 stations"
    )

    start_and_end_features = []
    with st.spinner(text="Fetching features to be used for plotting"):
        
        for scenario in config.displayed_scenario_names.keys():
            inference = InferenceModule(scenario=scenario)
            features = inference.fetch_time_series_and_make_features(target_date=datetime.now(UTC), geocode=False)

            # Add station names to features
            ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
            features[f"{scenario}_station_name"] = features[f"{scenario}_station_id"].map(ids_and_names)
            start_and_end_features.append(features)


    with st.spinner(text="Fetching the geographical data and the predictions for the next hour"):

        geographical_features_and_predictions = pd.read_parquet(
            path=INFERENCE_DATA/"geographical_features_and_predictions.parquet"
        )
    
    with st.spinner(text=""):
        
        station_name = geographical_features_and_predictions["station_name"].iloc[row_id]
        
        for scenario in config.displayed_scenario_names.keys():
            row_indices = np.argsort(geographical_features_and_predictions[f"predicted_{scenario}s"].values)[::-1]
        
            for row_id in row_indices[:10]:
                prediction = geographical_features_and_predictions[f"predicted_{scenario}s"].iloc[row_id]

                st.metric(
                    label=f"Predicted {config.displayed_scenario_names[scenario].lower()}", 
                    value=int(prediction)
                )

                fig = plot_for_one_station(
                    scenario=scenario,
                    station_name=station_name,
                    features=features,
                    targets=geographical_features_and_predictions[[f"predicted_{scenario}s", "station_name"]],
                    predictions=pd.Series(geographical_features_and_predictions[f"predicted_{scenario}s"]),
                    display_title=True
                )

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)