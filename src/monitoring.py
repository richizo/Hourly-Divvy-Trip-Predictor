"""
This module contains code that creates a new feature view based on the feature groups
for features and predictions. This feature view will be referenced in order to get 
feature and prediction data for the purpose of monitoring model performance inside 
the streamlit frontend.
"""
import pandas as pd 
from datetime import datetime

from src.setup.config import config
from src.inference_pipeline.feature_store_api import FeatureStoreAPI


def load_predictions_and_historical_trips(
    scenario: str,
    model_name: str,
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """


    Args:
        scenario (str): _description_
        model_name (str): _description_
        from_date (datetime): _description_
        to_date (datetime): _description_

    Returns:
        pd.DataFrame: the data to be used fo
    """
    tuned_or_not = "untuned"
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)

    arrivals_or_departures: str = config.displayed_scenario_names[scenario].lower()

    api = FeatureStoreAPI(
        scenario=scenario,
        api_key=config.hopsworks_api_key,
        project_name=config.hopsworks_project_name,
        event_time=None,
        primary_key=None
    )

    predictions_fg = api.setup_feature_group(
        description=f"predicting {arrivals_or_departures} - {tuned_or_not} {model_name}",
        name=f"{model_name}_{scenario}_predictions_feature_group",
        for_predictions=True,
        version=6
    )

    historical_fg = api.setup_feature_group(
        description=f"Hourly time series data for {arrivals_or_departures.lower()}",
        name=f"{scenario}_feature_group",
        version=config.feature_group_version,
        for_predictions=False
    )

    query = (
        predictions_fg
        .select_all()
        .join(
            sub_query=historical_fg.select(features=[f"{scenario}_station_id", f"{scenario}_hour", "trips"]),
            on=[f"{scenario}_station_id", f"{scenario}_hour"]
        )
        .filter(predictions_fg[f"{scenario}_hour"] >= from_ts)
        .filter(predictions_fg[f"{scenario}_hour"] <= to_ts)
    )

    monitoring_feature_view = api.get_or_create_feature_view(
        feature_group=historical_fg,
        name=f"monitoring_feature_view_for_{arrivals_or_departures.lower()}",
        use_sub_query=True,
        sub_query=query,
        version=1
    )

    monitoring_data = monitoring_feature_view.get_batch_data(start_time=from_date, end_time=to_date)

    return monitoring_data[
        monitoring_data[f"{scenario}_hour"].between(from_ts, to_ts)
    ]
