"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""


import numpy as np
import pandas as pd

from loguru import logger
from argparse import ArgumentParser

from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from sklearn.pipeline import Pipeline

from src.setup.config import FeatureGroupConfig, config

from src.feature_pipeline.feature_engineering import perform_feature_engineering
from src.inference_pipeline.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.model_registry_api import ModelRegistry


class InferenceModule:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.n_features = config.n_features

        self.feature_store_api = FeatureStoreAPI(
            scenario=self.scenario,
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            primary_key=["timestamp", f"{self.scenario}_station_id"],
            event_time="timestamp"
        )

        self.feature_group_metadata = FeatureGroupConfig(
            name=f"{scenario}_feature_group",
            version=config.feature_group_version,
            primary_key=self.feature_store_api.primary_key,
            event_time=self.feature_store_api.event_time
        )

        self.feature_group: FeatureGroup = self.feature_store_api.get_or_create_feature_group(
            name=self.feature_group_metadata.name,
            version=self.feature_group_metadata.version,
            description=f"Hourly time series data showing when trips {self.scenario}s"
        )

    def make_base_features(self, station_ids: list[int], ts_data: pd.DataFrame) -> pd.DataFrame:
        """
        Restructure 

        Args:
            station_ids: the list of unique station IDs
            ts_data: the time series data that is store on the feature store
            geocode: whether to implement geocoding during feature engineering

        Returns:
            pd.DataFrame: the dataframe consisting of the features
        """
        x = np.ndarray(
            shape=(len(station_ids), self.n_features), dtype=np.float64
        )

        for i, station_id in enumerate(station_ids):
            
            ts_data_i = ts_data.loc[
                ts_data[f"{self.scenario}_station_id"] == station_id, :
            ]

            ts_data_i = ts_data_i.sort_values(
                by=[f"{self.scenario}_hour"]
            )

            x[i, :] = ts_data_i["trips"].values

        base_features = pd.DataFrame(
            x, columns=[f"trips_previous_{i + 1}_hour" for i in reversed(range(self.n_features))]
        )

        return base_features

    def load_time_series_from_store(self, target_date: datetime) -> pd.DataFrame:
        """

        Args:
            target_date:

        Returns:
            pd.DataFrame: 
        """
        fetch_data_from = target_date - timedelta(days=28)
        fetch_data_to = target_date - timedelta(hours=1)

        feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view(
            name=f"{self.scenario}_feature_view",
            version=1,
            feature_group=self.feature_group
        )

        ts_data: pd.DataFrame = feature_view.get_batch_data(start_time=fetch_data_from, end_time=fetch_data_to)

        ts_data = ts_data.sort_values(
            by=[f"{self.scenario}_station_id", f"{self.scenario}_hour"]
        )
        
        station_ids = ts_data[f"{self.scenario}_station_id"].unique()
        # assert len(ts_data) == config.n_features * len(station_ids), \
        #    "The time series data is incomplete on the feature store. Please review the feature pipeline."
    
        base_features = self.make_base_features(station_ids=station_ids, ts_data=ts_data)

        # Include the {self.scenario}_hour column
        base_features[f"{self.scenario}_hour"] = target_date
        base_features[f"{self.scenario}_station_id"] = station_ids

        # Perform feature engineering (without geocoding) to complete the transformation of the time series data
        engineered_features = perform_feature_engineering(
            features=base_features,
            scenario=self.scenario,
            geocode=False
        )

        if engineered_features.empty:
            breakpoint()

        else:
            print(engineered_features.shape)
            breakpoint()

        return engineered_features.sort_values(
            by=[f"{self.scenario}_station_id"]
        )

    def load_predictions_from_store(
            self,
            model_name: str,
            from_hour: datetime,
            to_hour: datetime
    ) -> pd.DataFrame:
        """

        Args:
            model_name: the model's name is part of the name of the feature view to be queried
            from_hour:
            to_hour:

        Returns:

        """
        predictions_feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view(
            name=f"{model_name}_predictions_from_feature_store",
            version=1,
            feature_group=self.feature_group
        )

        logger.info(f'Fetching predictions for "{self.scenario}_hours" between {from_hour} and {to_hour}')
        predictions = predictions_feature_view.get_batch_data(start_time=from_hour, end_time=to_hour)

        predictions[f"{self.scenario}_hour"] = pd.to_datetime(predictions[f"{self.scenario}_hour"], utc=True)
        from_hour = pd.to_datetime(from_hour, utc=True)
        to_hour = pd.to_datetime(to_hour, utc=True)

        predictions = predictions[
            predictions[f"{self.scenario}_hour"].between(from_hour, to_hour)
        ]

        predictions = predictions.sort_values(
            by=[f"{self.scenario}_hour", f"{self.scenario}_station_id"]
        )

        return predictions


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
        prediction_per_station[f"predicted_{self.scenario}s"] = predictions.round(decimals=0)

        return prediction_per_station
