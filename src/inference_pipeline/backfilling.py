import pandas as pd
from pathlib import Path

from loguru import logger

from src.setup.config import config
from src.setup.paths import TIME_SERIES_DATA, PARENT_DIR
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.feature_store_api import FeatureStoreAPI


def backfill_feature_store(scenario: str) -> None:
    """
    Upload the time series data to the feature store.

    Args:
        scenario: "start" or "end"

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

    processor = DataProcessor(year=config.year)
    file_path = TIME_SERIES_DATA/f"{scenario}s_ts.parquet"
    if Path(file_path).is_file():
        ts_data = pd.read_parquet(file_path)
        logger.success("Retrieved the time series data")
    else:
        logger.exception(f"There is no saved time series data for the {scenario}s of trips")
        ts_data = processor.make_training_data(for_feature_store=True, geocode=False)[0] if scenario == "start" \
            else processor.make_training_data(for_feature_store=True, geocode=False)[1]

    ts_data["timestamp"] = ts_data[f"{scenario}_hour"].astype(int) // 10**6  # Express in milliseconds
    ts_data = ts_data.drop(f"{scenario}_hour", axis=1)
    feature_group = api.get_or_create_feature_group()
    feature_group.insert(ts_data, write_options={"wait_for_job": True})  # Push time series data to the feature group


if __name__ == "__main__":
    for scenario in ["start", "end"]:
        backfill_feature_store(scenario=scenario)
