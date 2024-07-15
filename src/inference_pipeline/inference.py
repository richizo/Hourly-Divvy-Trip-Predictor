"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""


import numpy as np
import pandas as pd

from comet_ml import API
from loguru import logger
from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from sklearn.pipeline import Pipeline

from src.setup.config import FeatureGroupConfig, config
from src.inference_pipeline.feature_store_api import FeatureStoreAPI


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

    def make_features(self, station_ids: list[int], time_series_data: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            station_ids: the list of unique station IDs
            time_series_data: the time series data that is store on the feature store

        Returns:
            pd.DataFrame: the dataframe consisting of the features
        """
        x = np.ndarray(
            shape=(len(station_ids), self.n_features), dtype=np.float64
        )

        for i, station_id in enumerate(station_ids):
            ts_data_i = time_series_data.loc[
                time_series_data[f"{self.scenario}_station_id"] == station_id, :
            ]

            ts_data_i = ts_data_i.sort_values(
                by=[f"{self.scenario}_hour"]
            )

            ts_data_i[i, :] = ts_data_i["trips"].values

        features = pd.DataFrame(
            x,
            columns=[f"rides_previous_{i + 1}_hour" for i in reversed(range(self.n_features))]
        )

        return features

    def load_time_series_from_store(self, target_date: datetime) -> pd.DataFrame:
        """

        Args:
            target_date:

        Returns:

        """
        fetch_data_from = target_date - timedelta(days=28)
        fetch_data_to = target_date - timedelta(hours=1)

        feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view(
            name=self.feature_group_metadata.name,
            version=self.feature_group.version,
            feature_group=self.feature_group
        )

        ts_data: pd.DataFrame = feature_view.get_batch_data(start_time=fetch_data_from, end_time=fetch_data_to)
        ts_first_date = int(fetch_data_from.timestamp())
        ts_last_date = int(fetch_data_to.timestamp())

        ts_data = ts_data[
            ts_data["timestamp"].between(left=ts_first_date, right=ts_last_date)
        ]

        ts_data = ts_data.sort_values(
            by=[f"{self.scenario}_station_id", f"{self.scenario}_hour"]
        )
        
        # Check that the data fetched from the feature store contains no missing data.
        station_ids = ts_data[f"{self.scenario}_station_id"].unique()
        assert len(ts_data) == config.n_features * len(station_ids), \
            "The time series data is incomplete on the feature store. Please review the feature pipeline."

        features = self.make_features(station_ids=station_ids, time_series_data=ts_data)
        #  features[f"{self.scenario}_hour"] = target_date
        #  features[f"{self.scenario}_station_id"] = station_ids

        return features.sort_values(
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
            model_name:
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
