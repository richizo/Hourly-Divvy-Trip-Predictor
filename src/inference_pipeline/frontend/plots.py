import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.express as px 
from plotly.graph_objects import Figure

from datetime import timedelta
from src.setup.config import config
from src.setup.paths import MIXED_INDEXER


def get_station_name(station_id: int):

    with open(MIXED_INDEXER/"end_geodata.json", mode="r") as file:
        geodata = json.load(file)

    for station in geodata:
        if station["station_id"] == station_id:
            return station["station_name"]


def plot_one_sample(
    scenario: str,
    station_id: int,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    display_title: bool | None = True,
    targets: pd.Series | None = None
) -> Figure:
    """

    Args:
        scenario (str): _description_
        station_id (int): _description_
        features (pd.DataFrame): _description_
        predictions (pd.DataFrame): _description_
        display_title (str, optional): _description_. Defaults to None.
        targets (pd.Series | None, optional): _description_. Defaults to None.
    """
    station_features: pd.DataFrame = features[features[f"{scenario}_station_id"] == station_id]
    station_targets = targets[targets[f"{scenario}_station_id"] == station_id] if targets is not None else None

    columns_of_past_trips = [column for column in station_features.columns if column.startswith("trips_previous_")]
    cumulative_trips = [station_features[column] for column in columns_of_past_trips] + [station_targets]

    all_trip_dates = pd.date_range(
        start=station_features[f"{scenario}_hour"] - timedelta(hours=len(columns_of_past_trips)),
        end=station_features[f"{scenario}_hour"],
        freq="H" 
    )

    # Surely the whole hour column can't be here
    title = f"{config.displayed_scenario_names[scenario]} hour = {station_features[f"{scenario}_hour"]}, Station name \
        ={get_station_name(station_id=station_id)} if display_title else None"

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
        station_prediction = predictions[predictions[f"{scenario}_station_id"] == station_id]

        figure.add_scatter(
            x=all_trip_dates[-1:], 
            y=[station_prediction], 
            line_color="red",
            marker_symbol="x", 
            mode="markers", 
            name="Prediction"
        )

    return figure 


# if __name__ != "__main__":

#    with st.spinner(text="Plotting time series data"):
        
