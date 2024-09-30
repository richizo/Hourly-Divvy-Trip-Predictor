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
    load_local_geojson, prepare_df_of_local_geodata, make_geodataframe, reconcile_geodata
)

from src.inference_pipeline.frontend.predictions import (
    retrieve_predictions, retrieve_predictions_for_this_hour, get_predictions_per_station
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


def perform_colour_scaling(geo_dataframe: GeoDataFrame, predictions: pd.DataFrame) -> GeoDataFrame:
    """
    Feed each of the predictions arrivals and departures into the pseudocolour function in order to produce the colour
    scaling effect.

    Args:
        geo_dataframes: a list of which will contain the geodataframes for arrivals and departures.
        predictions: the dataframe of predictions obtained from the feature store.

    Returns:
        list[GeoDataFrame]: a list of dataframes consisting of the merged external geodata (from the shapefile) and 
                            the predictions from the feature store.
    """
    black, green = (0, 0, 0), (0, 255, 0)

    geo_dataframe = geo_dataframe.rename(columns={f"{scenario}_station_name": "station_name"})
    predictions = predictions.rename(columns={f"{scenario}_station_name": "station_name"})

    logger.info("Merging geographical details and predictions for ", config.displayed_scenario_names[scenario].lower())

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

    merged_data[f"{scenario}_coordinates"] = merged_data[f"{scenario}_coordinates"].apply(tuple)
    return merged_data
        

@st.cache_resource
def make_map(_geodataframe_and_predictions: pd.DataFrame) -> None:

    initial_view_state = pdk.ViewState(
        latitude=41.872866,
        longitude=-87.63363,
        zoom=12,
        max_zoom=20,
        pitch=45,
        bearing=0
    )

    layer = pdk.Layer(
        data=_geodataframe_and_predictions,
        type="ScatterplotLayer",    
        get_position=f"{scenario}_coordinates",
        get_fill_color=f"{scenario}_fill_colour",
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        get_radius=70,
        auto_highlight=True,
        pickable=True
    )

    if scenario == "start":
        tooltip = {
            "html": "<b>Station:</b> {station_name} <br /> <b>Predicted departures in the next hour:</b> {predicted_starts}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    else:
        tooltip = {
            "html": "<b>Station:</b> {station_name} <br /> <b>Predicted arrivals in the next hour:</b> {predicted_ends}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    map = pdk.Deck(
        layers=layer,
        initial_view_state=initial_view_state,
        map_style="mapbox://styles/mapbox/navigation-day-v1",
        tooltip=tooltip
    )

    st.pydeck_chart(pydeck_obj=map)


if __name__ != "__main__":
    tracker = ProgressTracker(n_steps=4)

    user_choice = st.selectbox(
        label="Would you like to view predictions for arrivals or departures?",
        placeholder="Select an option",
        options=["Arrivals", "Departures"]
    )

    for scenario in config.displayed_scenario_names.keys():
        if config.displayed_scenario_names[scenario] in user_choice:

            with st.spinner(text="Collecting station information"):
                geo_dataframe = make_geodataframe(scenario=scenario)
                tracker.next()

            with st.spinner(
                text=f"Fetching all predicted {config.displayed_scenario_names[scenario].lower()} from feature store"
            ):
                predictions = retrieve_predictions(scenario=scenario)

                predictions_this_hour = retrieve_predictions_for_this_hour(
                    scenario=scenario,
                    predictions=predictions,
                    from_hour=config.current_hour-timedelta(hours=1),
                    to_hour=config.current_hour
                )

                tracker.next()

            with st.spinner(text="Setting up ingredients for the map"):
                
                geographical_features_and_predictions = perform_colour_scaling(
                    geo_dataframe=geo_dataframe,
                    predictions=predictions_this_hour
                )

                tracker.next()
                
            with st.spinner(text="Generating map of the Chicago area"):
                make_map(_geodataframe_and_predictions=geographical_features_and_predictions)
                tracker.next()

    st.sidebar.write("✅ Map Drawn")
