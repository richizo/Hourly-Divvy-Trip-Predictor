"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""

from pathlib import Path 

import numpy as np
import pandas as pd


from loguru import logger
from argparse import ArgumentParser

from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from sklearn.pipeline import Pipeline

from src.setup.config import FeatureGroupConfig, config

from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import perform_feature_engineering
from src.inference_pipeline.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.model_registry_api import ModelRegistry


class InferenceModule:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.n_features = config.n_features

        self.api = FeatureStoreAPI(
            scenario=self.scenario,
            event_time="timestamp",
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            primary_key=[f"{self.scenario}_station_id", f"{self.scenario}_hour"],
        )

        self.feature_group_metadata = FeatureGroupConfig(
            name=f"{scenario}_feature_group",
            version=config.feature_group_version,
            primary_key=self.api.primary_key,
            event_time=self.api.event_time
        )

        self.feature_group: FeatureGroup = self.api.get_or_create_feature_group(
            description=f"Hourly time series data showing when trips {self.scenario}s",
            version=self.feature_group_metadata.version,
            name=self.feature_group_metadata.name,
            for_predictions=False
        )

    def fetch_time_series_and_make_features(self, target_date: datetime, geocode: bool) -> pd.DataFrame:
        """
        Queries the offline feature store for time series data within a certain timeframe, and creates features
        features from that data. We then apply feature engineering so that the data aligns with the features from
        the original training data.

        My initial intent was to fetch time series data the 28 days prior to the target date. However, the class
        method that I am using to convert said data into features requires a larger dataset to work (see the while 
        loop in the get_cutoff_indices method from the preprocessing module). So after some experimentation, I 
        decided to go with 168 days of prior time series data. I will look to play around this number in the future.

        Args:
            target_date: the date for which we seek predictions.
            geocode: whether to implement geocoding during feature engineering

        Returns:
            pd.DataFrame:
        """ 
        feature_view: FeatureView = self.api.get_or_create_feature_view(
            name=f"{self.scenario}_feature_view",
            feature_group=self.feature_group,
            version=1
        )

        logger.info("Fetching time series data from the offline feature store...")
        fetch_from = target_date - timedelta(days=168)
        ts_data: pd.DataFrame = feature_view.get_batch_data(start_time=fetch_from, end_time=target_date)

        ts_data = ts_data.sort_values(
            by=[f"{self.scenario}_station_id", f"{self.scenario}_hour"]
        )

        station_ids = ts_data[f"{self.scenario}_station_id"].unique()
        features = self.make_features(station_ids=station_ids, ts_data=ts_data, geocode=False)

        # Include the {self.scenario}_hour column and the IDs
        features[f"{self.scenario}_hour"] = target_date

        return features.sort_values(
            by=[f"{self.scenario}_station_id"]
        )

    def fetch_predictions_group(self, model_name: str) -> FeatureGroup:

        tuned_or_not = "tuned" if self.scenario == "start" else "untuned"

        return self.api.get_or_create_feature_group(
            description=f"predictions on {self.scenario} data using the {tuned_or_not} {model_name}",
            name=f"{model_name}_{self.scenario}_predictions_feature_group",
            for_predictions=True,
            version=6
        )

    def make_features(self, station_ids: list[int], ts_data: pd.DataFrame, geocode: bool) -> pd.DataFrame:
        """
        Restructure the time series data into features in a way that aligns with the features 
        of the original training data.

        Args:
            station_ids: the list of unique station IDs.
            ts_data: the time series data that is store on the feature store.

        Returns:
            pd.DataFrame: the dataframe consisting of the features
        """
        processor = DataProcessor(year=config.year, for_inference=True)

        # Perform transformation of the time series data with feature engineering
        return processor.transform_ts_into_training_data(
            ts_data=ts_data,
            geocode=geocode,
            scenario=self.scenario, 
            input_seq_len=config.n_features,
            step_size=24
        )

    def load_predictions_from_store(
            self,
            model_name: str,
            from_hour: datetime,
            to_hour: datetime
    ) -> pd.DataFrame:
        """
        Load a dataframe containing predictions from their dedicated feature group on the offline feature store.
        This dataframe will contain predicted values between the specified hours. 

        Args:
            model_name: the model's name is part of the name of the feature view to be queried
            from_hour: the first hour for which we want the predictions
            to_hour: the last hour for would like to receive predictions.

        Returns:
            pd.DataFrame: the dataframe containing predictions.
        """
        # Ensure these times are datatimes
        from_hour = pd.to_datetime(from_hour, utc=True)
        to_hour = pd.to_datetime(to_hour, utc=True)

        predictions_group = self.fetch_predictions_group(model_name=model_name)

        predictions_feature_view: FeatureView = self.api.get_or_create_feature_view(
            name=f"{model_name}_{self.scenario}_predictions",
            feature_group=predictions_group,
            version=1
        )

        logger.info(f'Fetching predictions for between {from_hour} and {to_hour}')
        predictions_df = predictions_feature_view.get_batch_data(
            start_time=from_hour - timedelta(days=1), 
            end_time=to_hour + timedelta(days=1)
        )

        print(
           f" There are {len(predictions_df[f"{self.scenario}_station_id"])} station ids from the feature view"
        )
        breakpoint()

        predictions_df[f"{self.scenario}_hour"] = pd.to_datetime(predictions_df[f"{self.scenario}_hour"], utc=True)

        return predictions_df.sort_values(
            by=[f"{self.scenario}_hour", f"{self.scenario}_station_id"]
        )

    def get_model_predictions(self, model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
        """
        Simply use the model's predict method to provide predictions based on the supplied features

        Args:
            model: the model object fetched from the model registry
            features: the features obtained from the feature store

        Returns:
            pd.DataFrame: the model's predictions
        """
        predictions = model.predict(features)
        prediction_per_station = pd.DataFrame()

        prediction_per_station[f"{self.scenario}_station_id"] = features[f"{self.scenario}_station_id"].values
        prediction_per_station[f"{self.scenario}_hour"] = pd.to_datetime(datetime.utcnow()).floor("H")
        prediction_per_station[f"predicted_{self.scenario}s"] = predictions.round(decimals=0)
        return prediction_per_station
