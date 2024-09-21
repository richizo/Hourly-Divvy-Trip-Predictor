"""
This experimental module contains code that:
 - loads geospatial data using shapefiles and makes it available for mapping
 - displays the locations of the various stations on an interactive map. It has been challenging to create an
 implementation of this that produces a good experience.
"""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import geopandas as gpd

from loguru import logger

from src.inference_pipeline.frontend.main import ProgressTracker
from src.inference_pipeline.frontend.data import ShapeFile, load_local_geojson, prepare_df_of_local_geodata

from src.inference_pipeline.frontend.predictions import (
    extract_predictions_for_this_hour, get_all_predictions, get_predictions_per_station
)

def remove_stations_with_no_predictions(
    scenario: str,
    external_geodata: gpd.GeoDataFrame,
    predictions: pd.DataFrame
) -> pd.DataFrame:

    stations_we_have_predictions_for = predictions[f"{scenario}_station_names"].unique()
    external_data_rows_with_predictions = np.isin(
        element=external_geodata["station_name"],
        test_elements=stations_we_have_predictions_for
    )

    num_unmapped_stations = len(external_geodata) - len(external_data_rows_with_predictions)
    logger.warning(
        f"{num_unmapped_stations} stations won't be plotted because we currently have no predictions for them"
    )

    # For plotting reasons, there's no need to keep shapefile data for stations that we don't have predictions for
    return external_geodata.loc[external_data_rows_with_predictions, :]


def pseudocolour(
    value: float,
    min_value: float,
    max_value: float,
    start_colour: tuple[int, int, int],
    stop_colour: tuple[int, int, int]
) -> tuple[float, ...]:
    """
    Use linear interpolation to convert a given value into a tuple(representing a colour) in the range between
    start_colour and stop_colour. Credit to https://stackoverflow.com/a/10907855

    Args:
        value: the input value to be converted into a tuple that represents a colour
        min_value: the smallest value in the range of available input values
        max_value: the largest value in the range of available input values
        start_colour: a tuple representing the RGB values of the colour on the extreme left of the colour scale
        stop_colour:a tuple representing the RGB values of the colour on the extreme right of the colour scale

    Returns:

    """
    relative_value = float(value-min_value)/(max_value-min_value)
    return tuple(
        relative_value*(b-a) + a for (a,b) in zip(start_colour, stop_colour)
    )


def perform_colour_scaling(
    scenario: str,
    external_geodata: gpd.GeoDataFrame,
    predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Feed each of the predicted arrivals and departures into the pseudocolour function in order to produce the colour
    scaling effect.

    Args:
        scenario: "start" or "end"
        external_geodata: geographical data obtained from the shapefile.
        predictions: the dataframe of predictions obtained from the feature store.

    Returns:
        pd.DataFrame: data consisting of the merged external geodata (from the shapefile) and the predictions from the
                      feature store.
    """
    merged_data = pd.merge(left=external_geodata, right=predictions, right_on=f"{scenario}_station_name")

    black, green = (0, 0, 0), (0, 255, 0)
    merged_data["colour_scaling"] = merged_data[f"predicted {scenario}s"]
    max_prediction, min_prediction = merged_data["colour_scaling"].max(), merged_data["colour_scaling"].min()

    merged_data["fill_colour"] = merged_data["colour_scaling"].apply(
        lambda x: pseudocolour(
            value=x,
            min_value=min_prediction,
            max_value=max_prediction,
            start_colour=black,
            stop_colour=green
        )
    )

    return merged_data


@st.cache_data
@st.cache_resource
def draw_map(geodata_and_predictions: pd.DataFrame):
    """

    Args:
        geodata_and_predictions (pd. DataFrame): _description_
    """
    with st.spinner("Building map..."):

        initial_view_state = pdk.ViewState(
            latitude=41.872866,
            longitude=-87.63363,
            zoom=11,
            max_zoom=20,
            pitch=45,
            bearing=0
        )

        geojson_layer = pdk.Layer(
            data=geodata_and_predictions,
            type="GeoJsonLayer",
            opacity=0.25,
            stroked=False,
            filled=True,
            extruded=False,
            wireframe=True,
            get_elevation=10,
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255],
            auto_highlight=True,
            pickable=True
        )

        tooltip = {"html": "<b>Zone:</b> [{station_name} <br /> <b>Predicted arrivals:</b> {predicted_ends}"}

if __name__ != "__main__":
    st.set_page_config(layout="wide")
    tracker = ProgressTracker(n_steps=6)
    shapefile = ShapeFile()

    with st.spinner(text=f"Fetching predicted arrivals and departures from feature store"):
        predicted_starts, predicted_ends = get_all_predictions()


    external_geodata = remove_stations_with_no_predictions(
        external_geodata=shapefile.load_data_from_shapefile(),
        predictions=
    )


    geojson = load_local_geojson(scenario="start")
    tracker.next()  # Keeping the progress bar code outside the execution of these cached functions

    geodata = prepare_df_of_local_geodata(scenario="start", geojson=geojson)
    tracker.next()


    st.sidebar.write("✅ Map Drawn")
    tracker.next()
