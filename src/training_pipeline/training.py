import pickle
import pandas as pd

from pathlib import Path
from loguru import logger
from comet_ml import Experiment
from argparse import ArgumentParser
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error

from src.feature_pipeline.feature_engineering import perform_feature_engineering

from src.setup.config import settings
from src.setup.paths import MODELS_DIR, TRAINING_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.training_pipeline.models import BaseModel, get_model
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparams


def get_or_make_training_data(scenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetches or builds the training data.

    Args:
        scenario (str): "start" or "end".

    Returns:
        pd.DataFrame: _description_
    """
    data_path = TRAINING_DATA/f"{scenario}s.parquet"
    if Path(data_path).is_file():
        training_data = pd.read_parquet(path=data_path)
        logger.success("The training data has already been created and saved. Fetched it...")
    else:
        logger.warning("No training data is stored. Creating the dataset will take a long time...")
        training_sets = DataProcessor(year=settings.year).make_training_data()
        training_data = training_sets[0] if scenario.lower() == "start" else training_sets[1]
        logger.success("Training data produced successfully")

    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)
    return features.sort_index(), target.sort_index()


def train(
        model_name: str,
        scenario: str,
        tune_hyperparameters: bool | None,
        hyperparameter_trials: int | None,
        save: bool = True,
        geocode: bool = False
) -> None:
    """
    The function first checks for the existence of the training data, and builds it if 
    it doesn't find the file. Then it checks for a saved model. If it doesn't find a model, 
    it will go on to build one, tune its hyperparameters, save the resulting model.

    Args:
        model_name (str): the name of the model to be trained

        scenario:   a string indicating whether we are training data on the starts or ends of trips.
                    The only accepted answers are "start" and "end"
        
        tune_hyperparameters (bool | None, optional): whether to tune hyperparameters or not.

        hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters

        save (bool): whether to save the model (locally and on CometML)

        geocode(bool): whether to geocode during feature engineering
    """
    experiment = Experiment(
        api_key=settings.comet_api_key,
        workspace=settings.comet_workspace,
        project_name=settings.comet_project_name
    )

    model_fn = get_model(model_name=model_name)
    experiment.add_tags(tags=[model_name, scenario])
    features, target = get_or_make_training_data(scenario=scenario)
    engineered_features = perform_feature_engineering(features=features, scenario=scenario, geocode=geocode)

    train_sample_size = int(0.9 * len(engineered_features))
    x_train, x_test = engineered_features[:train_sample_size], engineered_features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    if isinstance(model_fn, BaseModel):
        tune_hyperparameters = False
        hyperparameter_trials = None

    if not tune_hyperparameters:
        experiment.add_tag("Not tuned")
        logger.info("Using the default hyperparameters")
        model_fn = make_pipeline(model_fn)
    else:
        experiment.add_tag("Tuned")
        logger.info(f"Tuning hyperparameters of the {model_name} model. Have a snack and watch One Piece")

        best_model_hyperparams = optimise_hyperparams(
            model_fn=model_fn,
            hyperparameter_trials=hyperparameter_trials,
            scenario=scenario,
            experiment=experiment,
            x=x_train,
            y=y_train
        )

        logger.success(f"Best model hyperparameters {best_model_hyperparams}")
        model_fn = model_fn(**best_model_hyperparams)

    logger.info("Fitting model...")
    # The setup base model requires that we specify these parameters, whereas with one of the other models,
    # specifying the arguments causes an error.
    if isinstance(model_fn, Pipeline):
        model_fn.fit(X=x_train, y=y_train)
    else:
        model_fn.fit(x_train=x_train, y_train=y_train)

    y_pred = model_fn.predict(x_test)
    test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    experiment.log_metric(name="Test M.A.E", value=test_error)
    experiment.end()

    if save:
        tuned_or_not = "tuned" if tune_hyperparameters else "Not tuned"
        model_file_name = f"Best_{tuned_or_not}_{model_name}_model_for_{scenario}s.pkl"
        with open(MODELS_DIR/model_file_name, mode="wb") as file:
            pickle.dump(obj=model_fn, file=file)
        logger.success("Saved model to disk")


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
        hyperparameter_trials=args.hyperparameter_trials
    )
    