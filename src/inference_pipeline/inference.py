import pandas as pd

from datetime import datetime, timedelta
from hsfs.feature_view import FeatureView

from src.setup.config import FeatureGroupConfig, FeatureViewConfig, config
from src.inference_pipeline.feature_store_api import FeatureStoreAPI, create_hopsworks_api_object


class FeatureLoader:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.feature_store_api: FeatureStoreAPI = create_hopsworks_api_object(scenario=scenario)

#        self.feature_group_metadata = FeatureGroupConfig(
#            name=f"{self.scenario}_feature_group",
#            version=config.feature_group_version
#        )
#
#        self.feature_view_metadata = FeatureViewConfig(
#            name=f"{self.scenario}_feature_view",
#            version=config.feature_view_version
#        )

    def load_batch_of_features_from_store(self, target_date: datetime) -> pd.DataFrame:
        """

        Args:
            target_date:

        Returns:
            pd.DataFrame
        """
        fetch_data_from = target_date - timedelta(days=28)     
        fetch_data_to = target_date - timedelta(hours=1)
        feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view()

        ts_data = feature_view.get_batch_data(start_time=fetch_data_from, end_time=fetch_data_to)
        ts_start_date = int(ts_data["timestamp"])

