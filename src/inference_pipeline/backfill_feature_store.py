import pandas as pd
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser


from loguru import logger
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView


from src.setup.config import config
from src.setup.paths import TIME_SERIES_DATA, PARENT_DIR
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.model_registry_api import ModelRegistry
from src.inference_pipeline.inference import InferenceModule


class BackFiller:
    def __init__(self, scenario: str) -> None:
        """
        Introduce the scenario variable, and the Feature Store API object 

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
            primary_key=None,
            event_time=None
        )

    def backfill_features(self) -> None:
        """
        Upload the time series data to the feature store.

        Returns:
            None
        """
        self.api.primary_key = [f"timestamp", f"{self.scenario}_station_id"]
        self.api.event_time = "timestamp"

        processor = DataProcessor(year=config.year)
        file_path = TIME_SERIES_DATA / f"{self.scenario}s_ts.parquet"

        if Path(file_path).is_file():
            ts_data = pd.read_parquet(file_path)
            logger.success("Retrieved the time series data")
        else:
            logger.warning(
                f"There is no saved time series data for the {self.scenario}s of trips -> Building it..."
            )

            if self.scenario == "start":
                ts_data = processor.make_training_data(for_feature_store=True, geocode=False)[0]
            elif self.scenario == "end":
                ts_data = processor.make_training_data(for_feature_store=True, geocode=False)[1]

        ts_data["timestamp"] = ts_data[f"{scenario}_hour"].astype(int) // 10 ** 6  # Express in milliseconds

        #  ts_data = ts_data.drop(f"{scenario}_hour", axis=1)
        feature_group = self.api.get_or_create_feature_group(
            name=f"{self.scenario}_feature_group",
            version=config.feature_group_version,
            description=f"Hourly time series data showing when trips {self.scenario}"
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
        # The best model for (arrivals) departures was (un)tuned 
        tuned_or_not = "tuned" if self.scenario == "start" else "untuned"
        registry = ModelRegistry(scenario=self.scenario, model_name=model_name, tuned_or_not=tuned_or_not)
        model = registry.download_latest_model(status="production", unzip=True)

        inferrer = InferenceModule(scenario=self.scenario)
        features = inferrer.load_batch_of_features_from_store(target_date=target_date)
        predictions = inferrer.get_model_predictions(model=model, features=features)

        predictions_feature_group: FeatureGroup = self.api.get_or_create_feature_group(
            version=1,
            name=f"{model_name}_predictions_feature_group",
            description=f"predictions on {self.scenario} data using the {tuned_or_not} {model_name}."
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

        if args.target.lower() == "features":
            filler.backfill_features()

        elif args.target.lower() == "predictions":
            filler.backfill_predictions(target_date=datetime.now())
            
        else:
            raise Exception(
                'The only acceptable backfilling targets are "features" and "predictions"'
            )
    