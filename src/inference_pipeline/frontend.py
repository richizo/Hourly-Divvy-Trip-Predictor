import json
import pydeck
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd

from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

from src.setup.paths import GEOGRAPHICAL_DATA
from src.inference_pipeline.inference import InferenceModule


class Page:
    def __init__(self):
        self.n_steps = None
        self.progress_bar = None
        st.title("Divvy Trip Activity Predictor")

        self.current_date = pd.to_datetime(datetime.utcnow()).floor("H")
        st.header(f"{self.current_date} UTC")

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
    def _load_features_from_store(self, scenario: str, current_date: datetime) -> pd.DataFrame:
        """

        Args:
            current_date:

        Returns:
            pd.DataFrame
        """
        return InferenceModule(scenario=scenario).load_batch_of_features_from_store(target_date=current_date)

    @staticmethod
    @st.cache_data
    def _load_predictions_from_store(
            scenario: str,
            model_name: str,
            from_hour: datetime,
            to_hour: datetime
    ) -> pd.DataFrame:
        """

        Args:
            scenario:
            model_name:
            from_hour:
            to_hour:

        Returns:

        """
        inference = InferenceModule(scenario=scenario)
        return inference.load_predictions_from_store(from_hour=from_hour, to_hour=to_hour, model_name=model_name)

    @staticmethod
    def load_geodata(scenario: str) -> pd.DataFrame:
        with open(GEOGRAPHICAL_DATA / f"rounded_{scenario}_points_and_new_ids.geojson") as file:
            points_and_ids = json.load(file)
            geodata = pd.DataFrame(
                {"IDs": points_and_ids.keys(), "Coordinates": points_and_ids.values()}
            )

        return geodata

    def update_page_after_fetching_geodata_and_predictions(self, scenario: str, model_name: str):
        with st.spinner(text="Getting the coordinates of each station ID..."):
            geo_df = self.load_geodata(scenario=scenario)
            st.sidebar.write("✅ Coordinates obtained...")


        with st.spinner(text="Fetching model predictions from the feature store..."):
            predictions = self._load_predictions_from_store(
                scenario=scenario,
                model_name=model_name,
                from_hour=self.current_date - timedelta(hours=1),
                to_hour=self.current_date
            )

            st.write("✅ Model predictions received...")
            self.progress_bar.progress(2 / self.n_steps)

        next_hour_predictions_ready = \
            False if predictions[predictions[f"{scenario}_hour"] == self.current_date].empty else True

        previous_hour_predictions_ready = False if \
            predictions[predictions[f"{scenario}_hour"] == self.current_date - timedelta(hours=1)].empty else True

        if next_hour_predictions_ready:
            predictions = predictions[predictions[f"{scenario}_hour"] == self.current_date]
        elif previous_hour_predictions_ready:
            predictions = predictions[predictions[f"{scenario}_hour"] == self.current_date - timedelta(hours=1)]
            st.subheader("⚠️ Data from the current hour is not available. Using data from an hour ago.")
        else:
            raise Exception("Unable to find predictions for the current hour or the previous one.")

    @staticmethod
    def pseudocolour(value: int, min_value: int, max_value: int, start_colour: tuple, stop_colour: tuple):
        """
        Use linear interpolation to perform colour scaling on the predicted values. This provides us
        with a spectrum of colours for the prediction values.

        Credit to Pau Labarta Bajo and https://stackoverflow.com/a/10907855

        Args:
            value:
            min_value:
            max_value:
            start_colour:
            stop_colour:

        Returns:

        """

        f = float(
            (value - min_value) / (max_value - min_value)
        )

        return tuple(
            f * (b - a) + a for (a, b) in zip(start_colour, stop_colour)
        )

    def prep_data_for_plots(self, scenario: str, predictions: pd.DataFrame, geodata: pd.DataFrame) -> None:
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
            func=lambda x: self.pseudocolour(
                value=x,
                min_value=min_prediction,
                max_value=max_prediction,
                start_colour=black,
                stop_colour=green
            )
        )

        self.progress_bar.progress(3/self.n_steps)

    def make_map(self, geodata: pd.DataFrame) -> None:
        """

        Args:
            geodata:

        Returns:
            None
        """
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

        r = pydeck.Deck(
            layers=[geojson],
            initial_view_state=start_position,
            tooltip=tooltip
        )

        st.pydeck_chart(r)
        self.progress_bar.progress(4/self.n_steps)

    def fetch_features(self, scenario: str) -> None:
        """

        Args:
            scenario:

        Returns:
            None
        """
        with st.spinner(text="Getting a batch of features"):
            features = self._load_features_from_store(scenario=scenario, current_date=self.current_date)
            st.sidebar.write("✅ Features fetched from the store for inference")
            self.progress_bar.progress(5/self.n_steps)

    def plot_time_series(self, scenario: str, predictions: pd.DataFrame):

        with st.spinner(text="Plotting time series data..."):
            row_indices = np.argsort(predictions[f"predicted_{scenario}s"].values)[::-1]
            n_to_plot = 10

            for index in row_indices[:n_to_plot]:
                station_id = predictions[f"{scenario}_station_id"].iloc[index]

    def construct_page(self):
        menu_options = self.make_main_menu()

        if menu_options == "Plots":
            pass  # Just for now. Plotting logic will be provided later
        elif menu_options == "Predictions":
            self.progress_bar = st.sidebar.header("⚙️ Working Progress")
            self.progress_bar = st.sidebar.progress(value=0)
            self.n_steps = 7

            scenarios_and_choices = {"start": "Arrivals", "end": "Departures"}
            user_scenario_choice = st.sidebar.multiselect(
                label="Do you want predictions for the number of arrivals at or the departures from each station?",
                options=["Arrivals", "Departures"],
                placeholder="Please select one of the two options."
            )

            for scenario in scenarios_and_choices.keys():
                if scenarios_and_choices[scenario] in user_scenario_choice:
                    predictions = self._load_predictions_from_store(
                        scenario=scenario,
                        from_hour=self.current_date - timedelta(hours=1),
                        to_hour=self.current_date
                    )


if __name__ == "__main__":
    page = Page()
    page.construct_page()
