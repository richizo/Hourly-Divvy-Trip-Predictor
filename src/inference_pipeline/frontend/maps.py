"""
This experimental module contains code that:
 - displays the locations of the various stations on an interactive map. It has been challenging to create an 
   implementation of this that produces a good experience.
 - loads geospatial data using shapefiles and makes it available for mapping
"""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from loguru import logger
from datetime import timedelta
from geopandas import GeoDataFrame

from src.setup.config import config 
from src.inference_pipeline.frontend.main import ProgressTracker

from src.inference_pipeline.frontend.data import (
    load_local_geojson, prepare_df_of_local_geodata, make_geodataframes, reconcile_geodata
)

from src.inference_pipeline.frontend.predictions import (
    get_predictions_for_this_hour, get_all_predictions, get_predictions_per_station
)


def restrict_geodataframe_to_stations_with_predictions(
    scenario: str,
    predictions: pd.DataFrame,
    geo_dataframe: GeoDataFrame
) -> pd.DataFrame:
    """
    
    Args:
        scenario (str): _description_
        geo_dataframe (GeoDataFrame): _description_
        predictions (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    stations_we_have_predictions_for = predictions[f"{scenario}_station_name"].unique()

    predictions_are_present = np.isin(
        element=geo_dataframe[f"{scenario}_station_name"],
        test_elements=stations_we_have_predictions_for
    )

    number_present = [boolean for boolean in predictions_are_present if boolean == True]

    logger.warning(
        f"{len(geo_dataframe) - len(number_present)} stations won't be plotted due to a lack of predictions"
    )

    return geo_dataframe.loc[predictions_are_present, :]


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
    geo_dataframes: list[GeoDataFrame],
    predicted_starts: pd.DataFrame,
    predicted_ends: pd.DataFrame
) -> list[GeoDataFrame]:
    """
    Feed each of the predicted arrivals and departures into the pseudocolour function in order to produce the colour
    scaling effect.

    Args:
        geo_dataframes: a list of which will contain the geodataframes for arrivals and departures.
        predictions: the dataframe of predictions obtained from the feature store.

    Returns:
        list[GeoDataFrame]: a list of dataframes consisting of the merged external geodata (from the shapefile) and 
                            the predictions from the feature store.
    """
    black, green = (0, 0, 0), (0, 255, 0)
    geo_dataframes_merged_with_predictions = []
    
    for geo_dataframe in geo_dataframes:
        scenario = "start" if "start_station_name" in geo_dataframe.columns else "end"
        predictions = predicted_starts if scenario == "start" else predicted_ends
        geo_dataframe = geo_dataframe.rename(columns={f"{scenario}_station_name": "station_name"})
        predictions = predictions.rename(columns={f"{scenario}_station_name": "station_name"})

        logger.info(f"Merging geographical details and predictions for {config.displayed_scenario_names[scenario].lower()}")

        merged_data = pd.merge(left=geo_dataframe, right=predictions, left_on="station_name", right_on="station_name")
        merged_data[f"{scenario}_colour_scaling"] = merged_data[f"predicted_{scenario}s"]
        max_prediction, min_prediction = merged_data[f"{scenario}_colour_scaling"].max(), merged_data[f"{scenario}_colour_scaling"].min()
    
        merged_data[f"{scenario}_fill_colour"] = merged_data[f"{scenario}_colour_scaling"].apply(
            lambda x: pseudocolour(
                value=x,
                min_value=min_prediction,
                max_value=max_prediction,
                start_colour=black,
                stop_colour=green
            )
        )

        geo_dataframes_merged_with_predictions.append(merged_data)
    
    complete_merger = pd.merge(
        left=geo_dataframes_merged_with_predictions[0], 
        right=geo_dataframes_merged_with_predictions[1], 
        left_on="station_name", 
        right_on="station_name"
    )

    complete_merger = complete_merger.drop(
        ["0_x", "0_y", "geometry_y", "end_coordinates", "start_colour_scaling", "end_colour_scaling"], axis=1
    )

    complete_merger = complete_merger.rename(
        columns={"start_coordinates": "coordinates", "geometry_x": "geometry"}
    )

    complete_merger["coordinates"] = complete_merger["coordinates"].apply(tuple)
    return complete_merger


@st.cache_data
@st.cache_resource
def draw_map(_geodata_and_predictions: pd.DataFrame):
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
            data=_geodata_and_predictions["coordinates"],
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

        tooltip = {
            "html": "<b>Zone:</b> [{station_name} <br /> <b>Predicted departures:</b> {predicted_starts} \
                <b>Predicted arrivals:</b> {predicted_ends}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        map = pdk.Deck(
            layers=[geojson_layer],
            initial_view_state=initial_view_state,
            tooltip=tooltip
        )

        st.pydeck_chart(map)
        

if __name__ != "__main__":
    #st.set_page_config(layout="wide")
    tracker = ProgressTracker(n_steps=4)

    with st.spinner(text="Collecting station information"):
        start_geodataframe, end_geodataframe = make_geodataframes()
        tracker.next()

    with st.spinner(text="Fetching all predicted arrivals and departures from feature store"):
        all_predicted_starts, all_predicted_ends = get_all_predictions()

        predicted_starts_this_hour, predicted_ends_this_hour = get_predictions_for_this_hour(
            predicted_starts=all_predicted_starts,
            predicted_ends=all_predicted_ends,
            from_hour=config.current_hour-timedelta(hours=1),
            to_hour=config.current_hour
        )

    with st.spinner(text="Looking up the stations that we have predictions for"):
        start_geodataframe = restrict_geodataframe_to_stations_with_predictions(
            scenario="start", 
            predictions=predicted_starts_this_hour,
            geo_dataframe=start_geodataframe 
        )

        end_geodataframe = restrict_geodataframe_to_stations_with_predictions(
            scenario="end", 
            predictions=predicted_ends_this_hour,
            geo_dataframe=end_geodataframe
        )

        tracker.next()
     
    with st.spinner("Setting up ingredients for the map"):
        data_and_predictions = perform_colour_scaling(
            geo_dataframes=[start_geodataframe, end_geodataframe],
            predicted_starts=predicted_starts_this_hour,
            predicted_ends=predicted_ends_this_hour
        )

        tracker.next()
        breakpoint()

    with st.spinner(text="Generating map of the Chicago area"):
        draw_map(_geodata_and_predictions=data_and_predictions)
        tracker.next()

    st.sidebar.write("✅ Map Drawn")
    tracker.next()
