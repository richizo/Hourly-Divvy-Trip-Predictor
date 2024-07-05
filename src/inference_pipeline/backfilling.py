import pandas as pd
from src.setup.config import config

from src.training_pipeline.training import get_or_make_training_data
from src.inference_pipeline.feature_store_api import FeatureStoreAPI


def backfill_feature_store(scenario: str) -> None:
    """

    Args:
        scenario:

    Returns:

    """
    api = FeatureStoreAPI(
        scenario=scenario,
        api_key=config.hopsworks_api_key,
        project_name=config.hopsworks_project_name,
        feature_group_name=f"{scenario}_feature_group",
        feature_group_version=config.feature_group_version,
        feature_view_name=f"{scenario}_feature_view",
        feature_view_version=config.feature_view_version
    )

    features, target = get_or_make_training_data(scenario=scenario)

    feature_group = api.get_or_create_feature_group()
    feature_group.insert(
        features, write_options={"wait_for_job": True}
    )


if __name__ == "__main__":
    for scenario in ["start", "end"]:
        backfill_feature_store(scenario=scenario)
