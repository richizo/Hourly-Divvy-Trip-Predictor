"""
Contains code that:
- loads geographical data and predictions in order to feed the map.
- fetches predictions from the feature store so that it can be passed to the streamlit interface. 
- displays the locations of the various stations on an interactive map. It has been challenging to create an 
implementation of this that produces a good experience.
"""
import time 
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from loguru import logger
from datetime import datetime, timedelta
from streamlit_extras.colored_header import colored_header

from src.setup.config import config 
from src.setup.paths import FRONTEND_DATA, INFERENCE_DATA
from src.inference_pipeline.frontend.tracker import ProgressTracker
from src.inference_pipeline.backend.inference import load_predictions_from_store
from src.inference_pipeline.frontend.data import make_geodataframes, reconcile_geodata


@st.cache_data
def retrieve_predictions(from_hour: datetime, to_hour: datetime) -> pd.DataFrame:
    """ 
    Download all the predictions for all the stations from one hour to another

    Args:
        from_hour (datetime, optional): From which hour we want to fetch predictions. Defaults to the previous hour.
        to_hour (datetime, optional): the hour we want predictions for. Defaults to the current hour.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: a list of dataframes of predictions for both arrivals and departures
    """
    prediction_dataframes =[]
    for scenario in config.displayed_scenario_names.keys():                
        model_name = "lightgbm" if scenario == "end" else "xgboost"

        predictions: pd.DataFrame = load_predictions_from_store(
            scenario=scenario,
            model_name=model_name, 
            from_hour=from_hour, 
            to_hour=to_hour
        )
        
        prediction_dataframes.append(predictions)

    start_predictions, end_predictions = prediction_dataframes[0], prediction_dataframes[1]
    return start_predictions, end_predictions


@st.cache_data
def retrieve_predictions_for_this_hour(
    predicted_starts: pd.DataFrame,
    predicted_ends: pd.DataFrame,
    from_hour: datetime,
    to_hour: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Initialise an inference object, and load the dataframes of predictions which we already fetched from their 
    dedicated feature groups. Then fetch the most recent prediction if it is available, or the second most
    recent (the one from an hour before).

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
        pd.DataFrame: dataframes containing predicted arrivals and departures for this, or the previous hour.
    """
    all_predictions_this_hour = []
    scenario_and_predictions = {"start": predicted_starts, "end": predicted_ends}   

    for scenario in scenario_and_predictions.keys():
        predictions = scenario_and_predictions[scenario]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        next_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions[predictions[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            predictions_for_target_hour: pd.DataFrame = predictions[predictions[f"{scenario}_hour"] == to_hour]
            predictions_for_target_hour.to_parquet(path=INFERENCE_DATA/f"{scenario}_predictions.parquet")
        
        elif previous_hour_ready:
            predictions_for_target_hour = predictions[predictions[f"{scenario}_hour"] == from_hour]

            if scenario == "start":  # So this message only appears once
                st.write("⚠️ Predictions for the current hour are unavailable. Fetching those from an hour ago.")
        
        else:
            try:
                predictions_for_target_hour = pd.read_parquet(INFERENCE_DATA/f"{scenario}_predictions.parquet")
                if scenario == "start":
                    st.write("Unable to fetch predictions for this or the previous hour. Providing predictions from a while ago.")
            except:
                raise Exception("Oh no! Neither the online nor the offline sources of predictions are available.")


        # Now to include the names of stations
        predictions_for_target_hour  = predictions_for_target_hour.drop(f"{scenario}_station_id", axis = 1)
        predictions_for_target_hour = predictions_for_target_hour.reset_index(drop=True)
        all_predictions_this_hour.append(predictions_for_target_hour)

    start_predictions, end_predictions = all_predictions_this_hour[0], all_predictions_this_hour[1]
    return start_predictions, end_predictions


def restrict_geodataframe_to_stations_with_predictions(
    scenario: str,
    predictions: pd.DataFrame,
    geo_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Depending on how many days of time series data is fetched during inference (prior to applying the model for predictions),
    we may exclude certain stations from the dataframe of predictions (more is better in this regard). This function allows 
    us to eliminate such stations from the geographical data that we have before feeding this data into the map. That way, we
    don't display stations on the map if we don't have predictions for it.
    
    Args:
        scenario (str): "start" or "end"
        geo_dataframe (pd.DataFrame): the geographical data (either for arrivals or departures)
        predictions (pd.DataFrame): the dataframe of predictions fetched from the feature store (either for arrivals or departures)

    Returns:
        pd.DataFrame: geographical data for only those stations that we have predictions for
    """
    stations_we_have_predictions_for = predictions[f"{scenario}_station_name"].unique()
    predictions_are_present = np.isin(element=geo_dataframe[f"station_name"], test_elements=stations_we_have_predictions_for)
    number_present = [boolean for boolean in predictions_are_present if boolean == True]

    logger.warning(
        f"{len(geo_dataframe) - len(number_present)} stations won't be plotted because you only backfilled {config.backfill_days} days of predictions."
    )

    return geo_dataframe.loc[predictions_are_present, :]


def merge_geodataframe_and_predictions_per_scenario(scenario: str, geodataframe: pd.DataFrame, predictions: pd.DataFrame):

    predictions = predictions.rename(columns={f"{scenario}_station_name": "station_name"})
    return pd.merge(left=geodataframe, right=predictions, left_on="station_name", right_on="station_name")


class ColourModule:
    def __init__(self, value: int):
        self.red = (255, 0, 0)
        self.blue = (0,0,255)
        self.black = (0, 0, 0)
        self.value = value

    def pseudocolour(
        self,
        min_value: float,
        max_value: float
    ) -> tuple[float, ...]:
        """
        Use linear interpolation to convert a given value into a tuple(representing a colour) in the range between two 
        colours (one called the start_colour and the other called the stop_colour). The start_colour and stop_colour are 
        RGB triples which represent the colours on the extreme left and right of a specific scale respectively. Which 
        colour scale we are using will depend on the sign of the given value.

        Credit to https://stackoverflow.com/a/10907855.

        Args:
            value: the input value to be converted into a tuple that represents a colour
            min_value: the smallest value in the range of available input values
            max_value: the largest value in the range of available input values

        Returns:
            tuple[float...]: 
        """
        if self.value == 0:
            start_colour, stop_colour = self.black, self.black
        elif self.value > 0:
            start_colour, stop_colour = self.black, self.red
        elif self.value < 0:
            start_colour, stop_colour = self.black, self.blue

        relative_value =  float(self.value-min_value)/(max_value-min_value)
        shade = tuple(relative_value*(b-a) + a for (a,b) in zip(start_colour, stop_colour))
        return shade 


def colour_points_by_discrepancy(merged_data: pd.DataFrame) -> pd.DataFrame:

    
    merged_data["discrepancy"] = merged_data["predicted_starts"] - merged_data["predicted_ends"]
    
    merged_data[f"fill_colour"] = merged_data["discrepancy"].apply(
        lambda x: ColourModule(value=x).pseudocolour(
            min_value=merged_data["discrepancy"].min(),
            max_value=merged_data["discrepancy"].max()
        )
    )

    return merged_data
    

def fully_merge_data(
    start_geodataframe: pd.DataFrame,
    end_geodataframe: pd.DataFrame,
    predicted_starts: pd.DataFrame, 
    predicted_ends: pd.DataFrame
) -> pd.DataFrame:
    """
    Feed each of the predictions arrivals and departures into the pseudocolour function in order to produce the colour
    scaling effect.

    Args:
        geo_dataframes: a list of which will contain the geodataframes for arrivals and departures.
        predictions: the dataframe of predictions obtained from the feature store.

    Returns:
        list[pd.DataFrame]: a list of dataframes consisting of the merged external geodata (from the shapefile) and 
                            the predictions from the feature store.
    """
    geographical_features_and_predictions = []
    scenarios_and_geodataframes = {"start": start_geodataframe, "end": end_geodataframe}
    scenarios_and_predictions = {"start": predicted_starts, "end": predicted_ends}

    for scenario in config.displayed_scenario_names.keys():
        geo_dataframe = scenarios_and_geodataframes[scenario]
        predictions = scenarios_and_predictions[scenario]

        merged_data = merge_geodataframe_and_predictions_per_scenario(
            scenario=scenario,
            geodataframe=geo_dataframe, 
            predictions=predictions
        )

        merged_data[f"coordinates"] = merged_data[f"coordinates"].apply(tuple)
        geographical_features_and_predictions.append(merged_data)

    complete_merger = pd.merge(
        left=geographical_features_and_predictions[0], 
        right=geographical_features_and_predictions[1], 
        left_on=["station_name", "coordinates"], 
        right_on=["station_name", "coordinates"]
    )
    
    return colour_points_by_discrepancy(merged_data=complete_merger)

    
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
        get_position=f"coordinates",
        get_fill_color=f"fill_colour",
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        get_radius=50,
        auto_highlight=True,
        pickable=True
    )

    tooltip = {
        "style": {"backgroundColor": "green", "color": "white"},
        "html": "<b>{station_name} <br />\
                 <b>Predicted Arrivals:</b> {predicted_ends}<br /> <b>Predicted Departures:</b> {predicted_starts}"
    }

    map = pdk.Deck(
        layers=layer,
        initial_view_state=initial_view_state,
        map_style="mapbox://styles/mapbox/navigation-day-v1",
        tooltip=tooltip
    )

    st.pydeck_chart(pydeck_obj=map)


if __name__ != "__main__":

    tracker = ProgressTracker(n_steps=5)

    colored_header(
        label=":green[Hourly] :violet[Predictions]",
        description="",
        color_name="violet-70"
    )
    
    st.markdown(
        """
        Here you can view a map of the city centre and its environs with many :black[Divvy stations] littered all over 
        it. If you pan over to each station, you will be able to see the station's name (or at least, its address), and
        the number of :green[arrivals] and :orange[departures] :violet[predicted] to take place there :green[in the next hour].

        This process should only take about a minute.
        """
    )

    with st.spinner(text="Collecting station information"):
        start_geodataframe, end_geodataframe = make_geodataframes()
        tracker.next()

    with st.spinner(text=f"Fetching all predictions from the offline feature store"):
        predicted_starts, predicted_ends = retrieve_predictions(
            from_hour=config.current_hour - timedelta(hours=1),
            to_hour=config.current_hour
        )

        predicted_starts_this_hour, predicted_ends_this_hour = retrieve_predictions_for_this_hour(
            predicted_starts=predicted_starts,
            predicted_ends=predicted_ends,
            from_hour=config.current_hour-timedelta(hours=1),
            to_hour=config.current_hour
        )

        tracker.next()

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

    with st.spinner(text="Setting up ingredients for the map"):
        
        geographical_features_and_predictions = fully_merge_data(
            start_geodataframe=start_geodataframe,
            end_geodataframe=end_geodataframe,
            predicted_starts=predicted_starts_this_hour,
            predicted_ends=predicted_ends_this_hour
        )

        tracker.next()

    with st.spinner(text="Generating map of the stations in the Greater Chicago"):
        make_map(_geodataframe_and_predictions=geographical_features_and_predictions)
        geographical_features_and_predictions.to_parquet(FRONTEND_DATA/"geographical_features_and_predictions.parquet")
        tracker.next()

    st.sidebar.write("✅ Map Drawn")
