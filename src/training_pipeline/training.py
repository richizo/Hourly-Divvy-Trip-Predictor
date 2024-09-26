"""
Contains code for model training with and without hyperparameter tuning, as well as 
experiment tracking.
"""
import pickle
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from loguru import logger

from comet_ml import Experiment

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

from src.setup.config import config
from src.setup.paths import TRAINING_DATA, LOCAL_SAVE_DIR, make_fundamental_paths
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.training_pipeline.models import get_model
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparameters


class Trainer:
    def __init__(
        self,
        scenario: str,
        hyperparameter_trials: int,
        tune_hyperparameters: bool | None = True
    ):
        """
        Args:
            scenario (str): a string indicating whether we are training data on the starts or ends of trips.
                            The only accepted answers are "start" and "end"

            tune_hyperparameters (bool | None, optional): whether to tune hyperparameters or not.

            hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters
        """
        self.scenario = scenario
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_trials = hyperparameter_trials
        self.tuned_or_not = "Tuned" if self.tune_hyperparameters else "Untuned"
        make_fundamental_paths()  # Ensure that all the necessary directories exist.

    def get_or_make_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fetches or builds the training data for the starts or ends of trips.

        Returns:
            pd.DataFrame: a tuple containing the training data's features and targets
        """
        assert self.scenario.lower() in ["start", "end"]
        data_path = TRAINING_DATA / f"{self.scenario}s.parquet"
        
        if Path(data_path).is_file():
            training_data = pd.read_parquet(path=data_path)
            logger.success(f"Fetched saved training data for {config.displayed_scenario_names[self.scenario].lower()}")
        else:
            logger.warning("No training data is stored. Creating the dataset will take a while. Watch some One Piece.")

            processor = DataProcessor(year=config.year, for_inference=False)
            training_sets = processor.make_training_data(geocode=False)
            training_data = training_sets[0] if self.scenario.lower() == "start" else training_sets[1]
            
            logger.success("Training data produced successfully")

        target = training_data["trips_next_hour"]
        features = training_data.drop("trips_next_hour", axis=1)
        return features.sort_index(), target.sort_index()

    def train(self, model_name: str) -> float:
        """
        The function first checks for the existence of the training data, and builds it if
        it doesn't find it locally. Then it checks for a saved model. If it doesn't find a model,
        it will go on to build one, tune its hyperparameters, save the resulting model.

        Args:
            model_name (str): the name of the model to be trained

        Returns:
            float: the error of the chosen model on the test dataset.
        """
        model_fn: callable = get_model(model_name=model_name)
        features, target = self.get_or_make_training_data()

        train_sample_size = int(0.9 * len(features))
        x_train, x_test = features[:train_sample_size], features[train_sample_size:]
        y_train, y_test = target[:train_sample_size], target[train_sample_size:]

        experiment = Experiment(
            api_key=config.comet_api_key,
            workspace=config.comet_workspace,
            project_name=config.comet_project_name
        )
        
        if not self.tune_hyperparameters:
            experiment.set_name(name=f"{model_name.title()}(Untuned) model for the {self.scenario}s of trips")
            logger.info("Using the default hyperparameters")

            if model_name == "base":
                pipeline = make_pipeline(
                    model_fn(scenario=self.scenario)
                )
            else:
                if isinstance(model_fn, XGBRegressor):
                    pipeline = make_pipeline(model_fn)
                else:
                    pipeline = make_pipeline(model_fn())
        else:
            experiment.set_name(name=f"{model_name.title()}(Tuned) model for the {self.scenario}s of trips")
            logger.info(
                f"Tuning hyperparameters of the {model_name} model. Have a snack and watch One Piece (it's fantastic)"
            )

            best_model_hyperparameters = optimise_hyperparameters(
                model_fn=model_fn,
                hyperparameter_trials=self.hyperparameter_trials,
                experiment=experiment,
                x=x_train,
                y=y_train
            )

            logger.success(f"Best model hyperparameters {best_model_hyperparameters}")
            
            pipeline = make_pipeline(
                model_fn(**best_model_hyperparameters)
            )

        logger.info("Fitting model...")

        pipeline.fit(X=x_train, y=y_train)
        y_pred = pipeline.predict(x_test)
        test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)

        self.save_model_locally(model_fn=pipeline, model_name=model_name)
        experiment.log_metric(name="Test M.A.E", value=test_error)
        experiment.end()
        
        return test_error

    def save_model_locally(self, model_fn: Pipeline, model_name: str):
        """
        Save the trained model locally as a .pkl file

        Args:
            model_fn (Pipeline): the model object to be stored
            model_name (str): the name of the model to be saved
        """
        model_file_name = f"{model_name.title()} ({self.tuned_or_not} for {self.scenario}s).pkl"
        with open(LOCAL_SAVE_DIR/model_file_name, mode="wb") as file:
            pickle.dump(obj=model_fn, file=file)

        logger.success("Saved model to disk")

    def train_and_register_models(self, model_names: list[str], version: str, status: str) -> None:
        """
        Train the named models, identify the best performer (on the test data) and
        register it to the CometML model registry.

        Args:
            model_names: the names of the models under consideration
            version: the version of the registered model on CometML.
            status:  the registered status of the model on CometML.
        """
        models_and_errors = {}
        assert status.lower() in ["staging", "production"], 'The status must be either "staging" or "production"'
        
        for model_name in model_names:
            test_error = self.train(model_name=model_name)
            models_and_errors[model_name] = test_error

        test_errors = models_and_errors.values()
        for model_name in model_names:
            if models_and_errors[model_name] == min(test_errors):
                logger.info(f"The best performing model is {model_name} -> Pushing it to the CometML model registry")
                registry = ModelRegistry(model_name=model_name, scenario=self.scenario, tuned_or_not=self.tuned_or_not)
                registry.push_model_to_registry(status=status.title(), version=version)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--tune_hyperparameters", action="store_true")
    parser.add_argument("--hyperparameter_trials", type=int, default=15)
    args = parser.parse_args()

    trainer = Trainer(
        scenario=args.scenario,
        tune_hyperparameters=args.tune_hyperparameters,
        hyperparameter_trials=args.hyperparameter_trials
    )

    trainer.train_and_register_models(model_names=args.models, version="1.0.0", status="production")
