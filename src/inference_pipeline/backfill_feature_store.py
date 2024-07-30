import pandas as pd
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from loguru import logger
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from src.setup.config import config
from src.setup.paths import TIME_SERIES_DATA, PARENT_DIR, INFERENCE_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.inference_pipeline.inference import InferenceModule


class BackFiller:
    def __init__(self, scenario: str) -> None:
        """
        Introduce the scenario variable, and the Feature Store API object.

        Args:
            scenario: Determines whether we are looking at arrival or departure data. 
                      Its value must be "start" or "end".
        """
        self.scenario = scenario
        assert self.scenario.lower() in ["start", "end"], 'Only "start" or "end" are acceptable values'
        
        self.api = FeatureStoreAPI(
            scenario=self.scenario,
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            event_time=None,
            primary_key=None
        )

    def backfill_features(self) -> None:
        """
        Upload the time series data to the feature store.

        Returns:
            None
        """
        self.api.event_time = "timestamp"
        self.api.primary_key = ["timestamp", f"{self.scenario}_station_id"]
        ts_data_path = TIME_SERIES_DATA/f"{self.scenario}s_ts.parquet"

        if Path(ts_data_path).is_file():
            ts_data = pd.read_parquet(ts_data_path)
            logger.success("Retrieved the time series data")
        else:
            processor = DataProcessor(year=config.year)
            logger.warning(f"There is no saved time series data for the {self.scenario}s of trips -> Building it...")

            ts_data = processor.make_time_series()[0] if self.scenario.lower() == "start" else \
                processor.make_time_series()[1]

        ts_data["timestamp"] = ts_data[f"{scenario}_hour"].astype(int) // 10 ** 6  # Express in milliseconds

        #  ts_data = ts_data.drop(f"{scenario}_hour", axis=1)
        feature_group = self.api.get_or_create_feature_group(
            description=f"Hourly time series data showing when trips {self.scenario}",
            name=f"{self.scenario}_feature_group",
            version=config.feature_group_version
        )

        # Push time series data to the feature group
        feature_group.insert(
            ts_data,
            write_options={"wait_for_job": True}
        )

    def backfill_predictions(self, target_date: datetime, model_name: str = "lightgbm") -> None:
        """
        Fetch the registered version of the named model, and download it. Then load a batch of features
        from the relevant feature group(whether for arrival or departure data), and make predictions on those 
        features using the model. Then create or fetch a feature group for these predictions and push these  
        predictions. 

        Args:
            target_date (datetime): the date up to which we want our predictions.
            
            model_name (str, optional): the shorthand name of the model we will use. Defaults to "lightgbm",
                                        because the best performing models (for arrivals and departures) were
                                        LGBMRegressors.
        """
        self.api.primary_key = [f"{self.scenario}_station_id"]

        # The best model for (arrivals) departures was (un)tuned
        tuned_or_not = "tuned" if self.scenario == "start" else "untuned"
        
        inferrer = InferenceModule(scenario=self.scenario)
        registry = ModelRegistry(scenario=self.scenario, model_name=model_name, tuned_or_not=tuned_or_not)
        model = registry.download_latest_model(status="production", unzip=True)

        local_feature_path = INFERENCE_DATA/f"{self.scenario}s.parquet"

        if Path(local_feature_path).is_file():
            engineered_features = pd.read_parquet(local_feature_path)
        else:
            engineered_features = inferrer.fetch_time_series_and_make_features(
                target_date=datetime.now(),
                geocode=False
            )

        predictions = inferrer.get_model_predictions(model=model, features=engineered_features)

        predictions_feature_group: FeatureGroup = self.api.get_or_create_feature_group(
            description=f"predictions on {self.scenario} data using the {tuned_or_not} {model_name}",
            name=f"{model_name}_{self.scenario}_predictions_feature_group",
            version=1
        )

        # Push predictions to the feature group
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
        filler = BackFiller(scenario=scenario)
        logger.info(f"Working on the {scenario}s of trips...")

        if args.target.lower() == "features":
            filler.backfill_features()
        elif args.target.lower() == "predictions":
            filler.backfill_predictions(target_date=datetime.now())
        else:
            raise Exception('The only acceptable backfilling targets are "features" and "predictions"')
    