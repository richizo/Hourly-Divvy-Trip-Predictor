import json
from typing import Any

import pydeck
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

from datetime import datetime, timedelta, UTC
from streamlit_option_menu import option_menu
from shapely.geometry import Point 

from src.plot import plot_one_sample
from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


displayed_scenario_names = {"start": "Departures", "end": "Arrivals"}

st.title("Hourly Trip Predictor for Chicago's Divvy Bikes System")

current_hour = pd.to_datetime(datetime.now(UTC)).floor("H")
from_hour = current_hour-timedelta(hours=1)

st.header(f"{current_hour} UTC")

progress_bar = st.sidebar.header("⚙️ Working Progress")
progress_bar = st.sidebar.progress(value=0)

n_steps = 7


user_scenario_choice: list[str] = st.sidebar.multiselect(
    label="Do you want predictions for the number of arrivals at or the departures from each station?",
    options=["Arrivals", "Departures"],
    placeholder="Please select one of the two options."
)

for scenario in displayed_scenario_names.keys():
    if displayed_scenario_names[scenario] in user_scenario_choice:

        with st.spinner(text="Getting the coordinates of each station..."):

            with open(INDEXER_TWO/f"{scenario}_geodata.json") as file:
                geodata_dict = json.load(file)

            st.sidebar.write("✅ Retrieved Station Names, IDs & Coordinates")


        with st.spinner("Forming geodata"):

            coordinates = [tuple(value[0]) for value in geodata_dict.values()]
            station_ids = [value[1] for value in geodata_dict.values()]

            data = pd.DataFrame(
                data={
                    f"{scenario}_station_name": geodata_dict.keys(),
                    f"{scenario}_station_id": station_ids,
                    "coordinates": coordinates
                }
            )

            data["coordinates"] = data["coordinates"].apply(lambda x: Point(x[1], x[0]))
            geodata = gpd.GeoDataFrame(data, geometry="coordinates")
            geodata = geodata.set_crs(epsg=4326)

            
        with st.spinner(text="Fetching model predictions from the feature store..."):

            inferrer = InferenceModule(scenario=scenario)
            
            predictions_df: pd.DataFrame = inferrer.load_predictions_from_store(
                model_name="lightgbm",
                from_hour=from_hour, 
                to_hour=current_hour
            )

        next_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == current_hour].empty else True
        previous_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == current_hour]

        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == from_hour]
            current_hour = from_hour

        else:
            raise Exception("Cannot get predictions for either hour. The feature pipeline may not be working")

        if not predictions_to_use.empty:
            st.sidebar.write("✅ Retrieved Model Predictions")
            progress_bar.progress(2 / n_steps)


        with st.spinner(text="Preparing data for plotting..."):

            def color_scaling_map_locations(
                value: int, 
                min_value: int,
                max_value: int, 
                start_color: tuple, 
                stop_color: tuple
                ) -> tuple[float | Any]:
                """
                Use linear interpolation to perform color scaling on the predicted values. This provides us
                with a spectrum of colors for the prediction values.

                Credit to Pau Labarta Bajo and https://stackoverflow.com/a/10907855

                Args:
                    value (int): _description_
                    min_value (int): _description_
                    max_value (int): _description_
                    start_color (tuple): _description_
                    stop_color (tuple): _description_

                Returns:
                    tuple[float]: results of the interpolation
                """
                f = float(
                    (value - min_value) / (max_value - min_value)
                )

                return tuple(
                    f * (b - a) + a for (a, b) in zip(start_color, stop_color)
                )

            all_data = pd.merge(
                left=geodata,
                right=predictions_to_use,
                right_on=f"{scenario}_station_id",
                left_on=f"{scenario}_station_id",
                how="inner"
            )

            # Establish the max and min values as well as the start and stop colors for the color scaling.
            black, green = (0, 0, 0), (0, 255, 0)
            all_data["color_scaling"] = all_data[f"predicted_{scenario}s"]
            max_prediction, min_prediction = all_data["color_scaling"].max(), all_data["color_scaling"].min()

            # Perform color scaling
            all_data["fill_color"] = all_data["color_scaling"].apply(
                func=lambda x: color_scaling_map_locations(
                    value=x, 
                    min_value=min_prediction, 
                    max_value=max_prediction, 
                    start_color=black, 
                    stop_color=green
                )
            )

            progress_bar.progress(3 / n_steps)


        with st.spinner(text="Generating Map of Chicago"):

            # Selected a random coordinate to use as a start position
            start_position = pydeck.ViewState(latitude=41.872866, longitude=-87.63363, zoom=10, max_zoom=20, pitch=45, bearing=0)

            geojson = pydeck.Layer(
                type="GeoJsonLayer", 
                data=all_data,
                opacity=0.25,
                stroked=False,
                filled=True,
                extruded=False,
                get_elevation=10,
                get_fill_color="fill_color",
                get_line_color=[255, 255, 255],
                pickable=True   
            )

            tooltip = {"html": "<b>Station:</b> [{station_ID}]{station_name} <br /> <b>Predicted trips:</b> {predicted_trips}"}
            deck = pydeck.Deck(layers=[geojson], initial_view_state=start_position, tooltip=tooltip)

            st.pydeck_chart(pydeck_obj=deck)
            progress_bar.progress(4 / n_steps)


        with st.spinner(text="Getting a batch of features from the store..."):
            inferrer = InferenceModule(scenario=scenario)
            features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=False)

            st.sidebar.write("✅ Fetched features for inference")
            progress_bar.progress(5 / n_steps)


        with st.spinner(text="Plotting time series data..."):
            row_indices = np.argsort(predictions_to_use[f"predicted_{scenario}s"].values)[::-1]
            n_to_plot = 10

            for row_index in row_indices[:n_to_plot]:
                station_id = predictions_to_use[f"{scenario}_station_id"].iloc[row_index]
                prediction = predictions_to_use[f"predicted_{scenario}s"].iloc[row_index]

                st.metric(
                    label=f"Predicted {displayed_scenario_names[scenario]}", 
                    value=int(prediction)
                )

                fig = plot_one_sample(
                    scenario=scenario,
                    row_index=row_index,
                    features=features,
                    targets=predictions_to_use[f"predicted_{scenario}s"],
                    display_title=False
                )

                st.plotly_chart(figure_or_data=fig, theme="streamlit", use_container_width=True, width=1000)
            
            progress_bar.progress(6 / n_steps)
