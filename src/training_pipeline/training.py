import pickle
import subprocess
import pandas as pd

from pathlib import Path
from loguru import logger
from comet_ml import Experiment
from argparse import ArgumentParser
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from hsml.model_schema import ModelSchema, Schema

from src.setup.config import config
from src.setup.paths import MODELS_DIR, TRAINING_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.training_pipeline.models import BaseModel, get_model
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparameters

from src.inference_pipeline.feature_store_api import FeatureStoreAPI, create_hopsworks_api_object


def get_or_make_training_data(scenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetches or builds the training data for the starts or ends of trips.

    Args:
        scenario (str): "start" or "end".

    Returns:
        pd.DataFrame: a tuple containing the training data's features and targets
    """
    assert scenario.lower() == "start" or scenario.lower() == "end"
    data_path = TRAINING_DATA/f"{scenario}s.parquet"
    if Path(data_path).is_file():
        training_data = pd.read_parquet(path=data_path)
        logger.success("The training data has already been created and saved. Fetched it...")
    else:
        logger.warning("No training data is stored. Creating the dataset will take a long time...")
        training_sets = DataProcessor(year=config.year).make_training_data(for_feature_store=False, geocode=False)
        training_data = training_sets[0] if scenario.lower() == "start" else training_sets[1]
        logger.success("Training data produced successfully")

    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)
    return features.sort_index(), target.sort_index()


def provide_model_schema(x_train: pd.DataFrame, y_train: pd.Series) -> ModelSchema:
    input_schema = Schema(x_train)
    output_schema = Schema(y_train)
    return ModelSchema(input_schema=input_schema, output_schema=output_schema)


def train(
        model_name: str,
        scenario: str,
        tune_hyperparameters: bool | None,
        hyperparameter_trials: int | None,
        save_model_locally: bool = True,
        send_best_model_to_registry: bool = True
) -> float:
    """
    The function first checks for the existence of the training data, and builds it if
    it doesn't find it locally. Then it checks for a saved model. If it doesn't find a model,
    it will go on to build one, tune its hyperparameters, save the resulting model.

    Args:
        model_name (str): the name of the model to be trained

        scenario:   a string indicating whether we are training data on the starts or ends of trips.
                    The only accepted answers are "start" and "end"

        tune_hyperparameters (bool | None, optional): whether to tune hyperparameters or not.

        hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters

        save_model_locally (bool): whether to save the model locally.

        send_best_model_to_registry (bool): whether to send the best model (the one with the lowest error on the test
                                            set) to Hopsworks' model registry.
    """
    model_fn = get_model(model_name=model_name)
    features, target = get_or_make_training_data(scenario=scenario)

    train_sample_size = int(0.9 * len(features))
    x_train, x_test = features[:train_sample_size], features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    experiment = Experiment(
        api_key=config.comet_api_key, workspace=config.comet_workspace, project_name=config.comet_project_name
    )
    experiment.add_tags(tags=[model_name, scenario])

    if isinstance(model_fn, BaseModel):
        tune_hyperparameters = False
        hyperparameter_trials = None

    if not tune_hyperparameters:
        experiment.set_name(name=f"Untuned {model_name.title()} model for the {scenario}s of trips")
        logger.info("Using the default hyperparameters")
        if model_name == "base":
            pipeline = make_pipeline(model_fn(scenario=scenario))
        else:
            pipeline = make_pipeline(model_fn())

    else:
        experiment.set_name(name=f"Tuned {model_name.title()} model for the {scenario}s of trips")
        logger.info(f"Tuning hyperparameters of the {model_name} model. Have a snack and watch One Piece")

        best_model_hyperparameters = optimise_hyperparameters(
            model_fn=model_fn,
            hyperparameter_trials=hyperparameter_trials,
            experiment=experiment,
            x=x_train,
            y=y_train
        )

        logger.success(f"Best model hyperparameters {best_model_hyperparameters}")
        pipeline = make_pipeline(
            model_fn(**best_model_hyperparameters)
        )

    logger.info("Fitting model...")
    # The setup base model requires that we specify these parameters, whereas with one of the other models,
    # specifying the arguments causes an error.
    pipeline.fit(X=x_train, y=y_train)

    test_errors = []
    y_pred = pipeline.predict(x_test)
    test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    test_errors.append(test_error)

    experiment.log_metric(name="Test M.A.E", value=test_error)
    experiment.end()

    if save_model_locally:
        tuned_or_not = "tuned" if tune_hyperparameters else "Not tuned"
        model_file_name = f"Best_{tuned_or_not}_{model_name}_model_for_{scenario}s.pkl"
        with open(MODELS_DIR / model_file_name, mode="wb") as file:
            pickle.dump(obj=model_fn, file=file)
        logger.success("Saved model to disk")

    if send_best_model_to_registry:
        log_best_model_to_registry(
            scenario=scenario,
            test_error=test_error,
            model_name=model_name,
            x_train=x_train,
            y_train=y_train
        )

    return test_error


def log_best_model_to_registry(scenario: str, test_error: float, model_name: str, x_train, y_train) -> None:
    api = create_hopsworks_api_object(scenario=scenario)
    project = api.login_to_hopsworks()
    model_registry = project.get_model_registry()
    logger.success("Connected to Hopsworks' model registry")

    model = model_registry.sklearn.create_model(
        name=config.comet_project_name,
        metrics={"Test M.A.E": test_error},
        description=model_name,
        input_example=x_train.sample(),
        model_schema=provide_model_schema(x_train=x_train, y_train=y_train)
    )
    logger.success("Logged")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--model", type=str, default="lightgbm")
    parser.add_argument("--tune_hyperparameters", action="store_true")
    parser.add_argument("--hyperparameter_trials", type=int, default=15)
    args = parser.parse_args()

    train(
        model_name=args.model,
        scenario=args.scenario,
        tune_hyperparameters=args.tune_hyperparameters,
        hyperparameter_trials=args.hyperparameter_trials,
        save_model_locally=False,
        send_best_model_to_registry=True
    )

