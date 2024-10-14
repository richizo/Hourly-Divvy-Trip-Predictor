"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import json 
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from argparse import ArgumentParser

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER, INFERENCE_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names

from src.inference_pipeline.backend.inference import (
    load_raw_local_geodata, 
    fetch_predictions_group,
    fetch_time_series_and_make_features
)

from src.inference_pipeline.backend.feature_store_api import setup_feature_group
from src.inference_pipeline.backend.model_registry_api import ModelRegistry


def get_feature_group_for_time_series(scenario: str, primary_key: list[str], event_time: str):

    return setup_feature_group(
        scenario=scenario,
        primary_key=primary_key,
        event_time=event_time,
        description=f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}",
        name=f"{scenario}_feature_group",
        version=config.feature_group_version,
        for_predictions=False
    )


def backfill_features(scenario: str) -> None:
    """
    Run the preprocessing script and upload the time series data to the feature store.

    Args:
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

    Returns:
        None
    """
    event_time = "timestamp"
    primary_key = ["timestamp", f"{scenario}_station_id"]
    processor = DataProcessor(year=config.year, for_inference=False)
    ts_data = processor.make_time_series()[0] if scenario == "start" else processor.make_time_series()[1]
    ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms

    ts_feature_group = get_feature_group_for_time_series(scenario=scenario)

    # Push time series data to the feature group
    ts_feature_group.insert(
        ts_data,
        write_options={"wait_for_job": True}
    )


def backfill_predictions(
    scenario: str, 
    target_date: datetime, 
    local: bool = True, 
    using_mixed_indexer: bool = True
    ) -> None:
    """
    Fetch the registered version of the named model, and download it. Then load a batch of features
    from the relevant feature group(whether for arrival or departure data), and make predictions on those 
    features using the model. Then create or fetch a feature group for these predictions and push these  
    predictions. 

    Args:
        target_date (datetime): the date up to which we want our predictions.
        
    """
    primary_key = [f"{scenario}_station_id"]
    
    # Based on the best models for arrivals & departures at the moment
    model_name = "lightgbm" if scenario == "end" else "xgboost"
    tuned_or_not = "tuned" if scenario == "end" else "untuned"

    registry = ModelRegistry(scenario=scenario, model_name=model_name, tuned_or_not=tuned_or_not)
    model = registry.download_latest_model(unzip=True)
    ts_feature_group = get_feature_group_for_time_series(scenario=scenario, primary_key=primary_key, event_time=None)

    features = fetch_time_series_and_make_features(
        scenario=scenario,
        feature_group=ts_feature_group,
        start_date=target_date - timedelta(days=270),
        target_date=datetime.now(),
        geocode=False
    )
    
    try:
        features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
    except Exception as error:
        logger.error(error)    

    predictions: pd.DataFrame = get_model_predictions(scenario=scenario, model=model, features=features)
    predictions = predictions.drop_duplicates().reset_index(drop=True)

    # Now to add station names to the predictions
    ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
    predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

    if local:
        logger.warning(f"Logging predicted {config.displayed_scenario_names[scenario].lower()} locally")
        predictions.to_parquet(INFERENCE_DATA/f"{scenario}_predictions.parquet")
    else:    
        predictions_feature_group = setup_feature_group(
            description=f"predicting {config.displayed_scenario_names[scenario]} - {tuned_or_not} {model_name}",
            name=f"{model_name}_{scenario}_predictions_feature_group",
            for_predictions=True,
            version=6
        )

        predictions_feature_group.insert(
            predictions,
            write_options={"wait_for_job": True}
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        if args.target.lower() == "features":
            backfill_features(scenario=scenario)
        elif args.target.lower() == "predictions":
            backfill_predictions(scenario=scenario, target_date=datetime.now())
        else:
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
