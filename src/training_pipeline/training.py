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
    else: 
        logger.info("No training data is stored. Creating the dataset will take a long time...")
        training_sets = DataProcessor(year=settings.year)._make_training_data()
        training_data = training_sets[0] if scenario == "start" else training_sets[1]
        logger.info("Training data created successfully")
    return training_data.sort_index()


def train(
    model_name: str, 
    scenario: str, 
    tune_hyperparams: bool|None,
    hyperparam_trials: int|None
    ) -> None:
    """
    The function first checks for the existence of the training data, and builds it if 
    it doesn't find the file. Then it checks for a saved model. If it doesn't find a model, 
    it will go on to build one, tune its hyperparameters, save the resulting model.

    Args:
        model_name (str): the name of the model to be trained

        scenario -- a string indicating whether we are training on start or stop data. 
                    The only accepted answers are "start" and "stop"
        
        tune_hyperparams (bool | None, optional) -- whether to tune hyperparameters or not.
    """

    model_fn = get_model(model_name=model_name)

    experiment = Experiment(
        api_key=settings.comet_api_key,
        workspace=settings.comet_workspace,
        project_name=settings.comet_project_name
    )

    experiment.add_tag(tag=model_name)
    training_data = get_or_make_training_data(scenario=scenario)
    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)

    features = perform_feature_engineering(data=features, scenario=scenario, geocode=False)
    
    train_sample_size = int(0.9*len(features))
    X_train, X_test = features[:train_sample_size], features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    if not tune_hyperparams:
        experiment.add_tag("Tuned")
        logger.info("Using the default hyperparameters")
        pipeline = make_pipeline(model_fn())
    else:
        experiment.add_tag("Untuned")
        logger.info(
            f"Tuning hyperparameters of the {model_name} model. Have a snack and watch some One Piece"
        )

        best_model_hyperparams = optimise_hyperparams(
            model_fn=model_fn, 
            hyperparam_trials=hyperparam_trials, 
            scenario=scenario, 
            X=X_train, 
            y=y_train
        )

        logger.success(f"Best model hyperparameters {best_model_hyperparams}")

    logger.info("Fitting model")
    model_fn(**best_model_hyperparams).fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)

    logger.info(f"Test M.A.E: {test_error}")
    experiment.log_metric(name="Test M.A.E", value=test_error)

    logger.info("Saving model to disk")
    tuned_or_not = "tuned" if tune_hyperparams else "untuned"
    model_file_name = f"Best_{tuned_or_not}_{model_name}_model_for_{scenario}s.pkl" 
    
    with open(MODELS_DIR/model_file_name, mode="wb") as file:
        pickle.dump(file)
    
    logger.info("Logging model to CometML")
    experiment.log_model(name=model_name, file_or_folder=MODELS_DIR/model_file_name)


if __name__ == "__main__":
    train(model_name="lightgbm", scenario="start", tune_hyperparams=True, hyperparam_trials=5)