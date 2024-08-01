import json
import pydeck
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

from datetime import datetime, timedelta, UTC
from streamlit_option_menu import option_menu

from src.plot import plot_one_sample
from src.setup.config import config 
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoding
from src.inference_pipeline.inference import InferenceModule
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


class Page:
    def __init__(self):
        self.n_steps = None
        self.progress_bar = None
        self.current_hour = pd.to_datetime(datetime.now(UTC)).floor("H")
        self.displayed_scenario_names = {"start": "Departures", "end": "Arrivals"}
            
        st.title("Divvy Trip Activity Predictor")
        st.header(f"{self.current_hour} UTC")

    @staticmethod
    def make_main_menu():
        with st.sidebar:
            return option_menu(
                menu_title="Main Menu",
                options=["Plots", "Predictions"],
                menu_icon="list_nested",
                icons=["bar-chart-fill", "bullseye"]
            )

    @st.cache_data
    def provide_features(self, scenario: str, target_date: datetime) -> pd.DataFrame:
        """
        Initiate an inference object and use it to get features until the target date.
        features that we will use to fuel the model and produce predictions.

        Args:
            scenario (str): _description_
            target_date (datetime): _description_

        Returns:
            pd.DataFrame: the created (or fetched) features
        """
        with st.spinner(text="Getting a batch of features from the store..."):
            inferrer = InferenceModule(scenario=scenario)
            features = inferrer.fetch_time_series_and_make_features(target_date=target_date, geocode=False)

            st.sidebar.write("✅ Fetched features for inference")
            self.progress_bar.progress(5 / self.n_steps)
            return features 

    @st.cache_data
    def get_hourly_predictions(
        _self,
        scenario: str, 
        model_name: str, 
        from_hour: datetime,
        to_hour: datetime 
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

        inferrer = InferenceModule(scenario=scenario)

        predictions_df: pd.DataFrame = inferrer.load_predictions_from_store(
            model_name=model_name,
            from_hour=from_hour, 
            to_hour=to_hour
        )

        print(predictions_df[f"{scenario}_hour"])
        breakpoint()


        next_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == to_hour].empty else True
        previous_hour_ready = False if predictions_df[predictions_df[f"{scenario}_hour"] == from_hour].empty else True

        if next_hour_ready: 
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == to_hour]

        elif previous_hour_ready:
            st.subheader("⚠️ Predictions for the current hour are unavailable. Using those from an hour ago.")
            predictions_to_use = predictions_df[predictions_df[f"{scenario}_hour"] == from_hour]

        else:
            raise Exception(
                "Cannot get predictions for either hour. The feature pipeline may not be working"
            )

        return predictions_to_use

    @staticmethod
    def load_geodata(scenario: str, indexer: str = "two") -> pd.DataFrame | gpd.GeoDataFrame:
        """

        Args:
            scenario (str): _description_

        Returns:
            pd.DataFrame: 
        """

        if indexer == "one":

            with open(GEOGRAPHICAL_DATA / f"rounded_{scenario}_points_and_new_ids.geojson") as file:
                points_and_ids = json.load(file)

            loaded_geodata = pd.DataFrame(
                {
                    f"{scenario}_station_id": points_and_ids.keys(), 
                    "coordinates": points_and_ids.values()
                }
            )

            reverse_geocoding = ReverseGeocoding(scenario=scenario, geo_data=loaded_geodata)
            station_names_and_locations = reverse_geocoding.reverse_geocode()

            updated_geodata = reverse_geocoding.put_station_names_in_geodata(
                station_names_and_coordinates=station_names_and_locations
            )

            return updated_geodata
        
        elif indexer == "two":
            with open(INDEXER_TWO/f"{scenario}_geodata_indexer_two.json") as file:
                geodata_dict = json.load(file)

            coordinates = [value[0] for value in geodata_dict.values()]
            station_ids = [value[1] for value in geodata_dict.values()]

            geodata = gpd.GeoDataFrame(
                {
                    f"{scenario}_station_names": geodata_dict.keys(),
                    f"{scenario}_station_ids": station_ids,
                    "coordinates": coordinates
                }
            )

            return geodata
    
            
    @staticmethod
    def color_scaling_map_locations(
        value: int, 
        min_value: int,
        max_value: int, 
        start_colour: tuple, 
        stop_colour: tuple
        ) -> tuple[float]:
        """
        Use linear interpolation to perform colour scaling on the predicted values. This provides us
        with a spectrum of colours for the prediction values.

        Credit to Pau Labarta Bajo and https://stackoverflow.com/a/10907855

        Args:
            value (int): _description_
            min_value (int): _description_
            max_value (int): _description_
            start_colour (tuple): _description_
            stop_colour (tuple): _description_

        Returns:
            tuple[float]: _description_
        """
        
        f = float(
            (value - min_value) / (max_value - min_value)
        )

        return tuple(
            f * (b - a) + a for (a, b) in zip(start_colour, stop_colour)
        )

    def make_map(self, geodata: pd.DataFrame) -> None:
        """

        Args:
            geodata:

        Returns:
            None
        """
        # Selected a random coordinate to use as a start position
        start_position = pydeck.ViewState(
            latitude=41.872866,
            longitude=-87.63363,
            zoom=10,
            max_zoom=20,
            pitch=45,
            bearing=0
        )

        geojson = pydeck.Layer(type="GeoJsonLayer", data=geodata)
        tooltip = {"html": "<b>Zone:</b> [{StationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_trips}"}

        deck = pydeck.Deck(
            layers=[geojson],
            initial_view_state=start_position,
            tooltip=tooltip
        )

        st.pydeck_chart(pydeck_obj=deck)
        self.progress_bar.progress(4/self.n_steps)

    def prep_data_for_plotting(self, scenario: str, predictions: pd.DataFrame, geodata: pd.DataFrame) -> None:
        """

        Args:
            scenario: "start" or "end"
            predictions: the loaded predictions
            geodata: the dataframe of station IDs and coordinates

        Returns:
            None.
        """
        with st.spinner(text="Preparing data..."):
            data = pd.merge(
                left=geodata,
                right=predictions,
                right_on=f"{scenario}_station_id",
                left_on=f"{scenario}_station_id",
                how="inner"
            )

        # Establish the max and min values as well as the start and stop colours for the colour scaling.
        black, green = (0, 0, 0), (0, 255, 0)
        data["colour_scaling"] = data[f"predicted_{scenario}s"]
        max_prediction, min_prediction = data["colour_scaling"].max(), data["colour_scaling"].min()

        #  Perform color scaling
        data["fill_colour"] = data["colour_scaling"].apply(
            func=lambda x: self.color_scaling(
                value=x,
                min_value=min_prediction,
                max_value=max_prediction,
                start_colour=black,
                stop_colour=green
            )
        )

        self.progress_bar.progress(3 / self.n_steps)

    def plot_time_series(self, scenario: str, features: pd.DataFrame, predictions: pd.DataFrame):

        with st.spinner(text="Plotting time series data..."):
            row_indices = np.argsort(predictions[f"predicted_{scenario}s"].values)[::-1]
            n_to_plot = 10

            for row in row_indices[:n_to_plot]:
                station_id = predictions[f"{scenario}_station_id"].iloc[row]


                prediction = predictions[f"predicted_{scenario}s"].iloc[row]

                st.metric(
                    label=f"Predicted {self.displayed_scenario_names[scenario]}", 
                    value=int(prediction)
                )

                fig = plot_one_sample(
                    example_station=row,
                    features=features,
                    targets=predictions[f"predicted_{scenario}s"],
                    display_title=False
                )

                st.plotly_chart(figure_or_data=fig, theme="streamlit", use_container_width=True, width=1000)
            
            self.progress_bar.progress(6/self.n_steps)

    def construct_page(self, model_name: str):
        menu_options = self.make_main_menu()

        if menu_options == "Plots":
            pass  # Just for now. Plotting logic will be provided later
        elif menu_options == "Predictions":
            self.progress_bar = st.sidebar.header("⚙️ Working Progress")
            self.progress_bar = st.sidebar.progress(value=0)
            self.n_steps = 7

            user_scenario_choice: list[str] = st.sidebar.multiselect(
                label="Do you want predictions for the number of arrivals at or the departures from each station?",
                options=["Arrivals", "Departures"],
                placeholder="Please select one of the two options."
            )

            print(user_scenario_choice)
            breakpoint()

            for scenario in self.displayed_scenario_names.keys():
                if self.displayed_scenario_names[scenario] in user_scenario_choice:

                    with st.spinner(text="Getting the coordinates of each station..."):
                        geo_df = self.load_geodata(scenario=scenario)
                        st.sidebar.write("✅ Station IDs & Coordinates Obtained...")

                    with st.spinner(text="Fetching model predictions from the feature store..."):
                        predictions_df: pd.DataFrame = self.get_hourly_predictions(
                            scenario=scenario,
                            model_name=model_name,
                            from_hour=self.current_hour - timedelta(hours=1),
                            to_hour=self.current_hour
                        )

                        if not predictions_df.empty:
                            st.sidebar.write("✅ Dataframe containing the model's predictions received...")

                            # self is not hashable by the cacher. Made the prediction loader static and moved    line here.
                            self.progress_bar.progress(2 / self.n_steps)

if __name__ == "__main__":
    page = Page()
    page.construct_page(model_name="lightgbm")