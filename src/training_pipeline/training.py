import pickle
import pandas as pd

from pathlib import Path
from loguru import logger
from comet_ml import Experiment
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

from src.feature_pipeline.feature_engineering import perform_feature_engineering

from src.setup.config import settings
from src.setup.paths import MODELS_DIR, TRAINING_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.training_pipeline.models import get_model
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparams


def get_or_make_training_data(scenario: str) -> pd.DataFrame:
    """
    Fetches or builds the training data.

    Args:
        scenario (str): "start" or "end".

    Returns:
        pd.DataFrame: _description_
    """
    data_path = TRAINING_DATA/f"{scenario}s.parquet"
    if Path(data_path).is_file():
        logger.success("The training data has already been created and saved. Fetching it...")
        training_data = pd.read_parquet(path=data_path)
        #print(training_data.columns.get_loc("trips_previous_168_hour"))
        #breakpoint()
    else:
        logger.warning("No training data is stored. Creating the dataset will take a long time...")
        training_sets = DataProcessor(year=settings.year).make_training_data()
        training_data = training_sets[0] if scenario.lower() == "start" else training_sets[1]
        logger.success("Training data created successfully")
    return training_data.sort_index()


def train(
        model_name: str,
        scenario: str,
        tune_hyperparams: bool | None,
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
        
        tune_hyperparams (bool | None, optional): whether to tune hyperparameters or not.

        hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters

        save (bool): whether to save the model (locally and on CometML)

        geocode(bool): whether to geocode during feature engineering
    """

    experiment = Experiment(
        api_key=settings.comet_api_key, workspace=settings.comet_workspace, project_name=settings.comet_project_name
    )

    experiment.add_tag(tag=model_name)

    training_data = get_or_make_training_data(scenario=scenario)
    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)
    engineered_features = perform_feature_engineering(features=features, scenario=scenario, geocode=False)

    train_sample_size = int(0.9 * len(engineered_features))
    x_train, x_test = engineered_features[:train_sample_size], engineered_features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    model_fn = get_model(model_name=model_name)
    if not tune_hyperparams:
        experiment.add_tag("Tuned")
        logger.info("Using the default hyperparameters")
        pipeline = make_pipeline(model_fn)
    else:
        experiment.add_tag("Not tuned")
        logger.info(f"Tuning hyperparameters of the {model_name} model. Have a snack and watch One Piece")

        best_model_hyperparams = optimise_hyperparams(
            model_fn=model_fn,
            hyperparameter_trials=hyperparameter_trials,
            scenario=scenario,
            x=x_train,
            y=y_train
        )

        logger.success(f"Best model hyperparameters {best_model_hyperparams}")
        pipeline = make_pipeline(model_fn(**best_model_hyperparams))

    logger.info("Fitting model...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)

    logger.info(f"Test M.A.E: {test_error}")
    experiment.log_metric(name="Test M.A.E", value=test_error)

    if save:
        tuned_or_not = "tuned" if tune_hyperparams else "Not tuned"
        model_file_name = f"Best_{tuned_or_not}_{model_name}_model_for_{scenario}s.pkl"
        with open(MODELS_DIR/model_file_name, mode="wb") as file:
            pickle.dump(obj=pipeline, file=file)
        logger.success("Saved model to disk")

        experiment.log_model(name=model_name, file_or_folder=MODELS_DIR/model_file_name)
        logger.success("Logged model to CometML")


if __name__ == "__main__":
    train(model_name="xgboost", scenario="start", tune_hyperparams=True, hyperparameter_trials=5)
