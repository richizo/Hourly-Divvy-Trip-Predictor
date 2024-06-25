import optuna
import numpy as np
import pandas as pd

from loguru import logger
from comet_ml import Experiment
from optuna.samplers import TPESampler

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

from src.setup.config import settings
from src.training_pipeline.models import BaseModel
from src.feature_pipeline.feature_engineering import FeatureEngineering


def sampled_hyperparams(
        model_fn: BaseModel | Lasso | LGBMRegressor | XGBRegressor,
        trial: optuna.trial.Trial
) -> dict[str, str | int | float]:
    """
      Return the range of values of each hyperparameter under consideration.

      Returns:
          dict: the range of hyperparameter values to be considered
    """
    if model_fn == Lasso:
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 2.0, log=True)
        }

    elif model_fn == LGBMRegressor:
        return {
            "metric": "mean_absolute_error",
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 1),
            "importance_type": trial.suggest_categorical("importance_type", ["split", "gain"]),
            "subsample": trial.suggest_int("subsample", 0.1, 1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1)
        }

    elif model_fn == XGBRegressor:
        return {
            "objective": "reg:absoluteerror",
            "eta": trial.suggest_float("eta", 0.1, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "alpha": trial.suggest_float("alpha", 0, 2),
            "subsample": trial.suggest_int("subsample", 0.1, 1)
        }

    else:
        raise NotImplementedError("This model has not been implemented")


def optimise_hyperparams(
        model_fn: BaseModel | Lasso | LGBMRegressor | XGBRegressor,
        scenario: str,
        hyperparameter_trials: int,
        x: pd.DataFrame,
        y: pd.Series
) -> dict:
    """

  Args:
    model_fn: the model architecture to be used
    scenario: whether we are looking at the trip start or trip stop data.
    hyperparameter_trials: the number of optuna trials that will be run per mode scenario
    x: the dataframe of features
    y: the pandas series which contains the target variable

  Returns:

  """
    assert model_fn in [Lasso, LGBMRegressor, XGBRegressor]
    models_and_tags: dict[Lasso | LGBMRegressor | type, str] = {
        Lasso: "Lasso",
        LGBMRegressor: "LGBMRegressor",
        XGBRegressor: "XGBRegressor"
    }

    model_name = models_and_tags[model_fn]

    def objective(trial: optuna.trial.Trial) -> float:
        """
    Perform Time series cross validation, fit a pipeline to it (equipped with the)
    selected model, and return the average error across all cross validation splits.
    
    Args:
        trial: The optuna trial that's being optimised.
    
    Returns:
        float: Average error per split.
    """
        error_scores = []
        hyperparameters = sampled_hyperparams(model_fn=model_fn, trial=trial)
        tss = TimeSeriesSplit(n_splits=5)
        logger.info(f"Starting Trial {trial.number}")

        feature_eng = FeatureEngineering(scenario=scenario, features=x)
        # Use TSS to split the features and target variables for training and validation
        for split_number, (train_indices, val_indices) in enumerate(tss.split(x)):
            logger.info(f"Performing split number {split_number}")
            x_train, x_val = x.iloc[train_indices], x.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            logger.info("Fitting model after feature engineering")
            pipeline = make_pipeline(
                feature_eng.get_pipeline(geocode=False),
                model_fn(**hyperparameters)
            )
            pipeline.fit(x_train, y_train)

            logger.info("Evaluating the performance of the trial...")
            y_pred = model_fn.predict(x_val)
            error = mean_absolute_error(y_val, y_pred)
            error_scores.append(error)
            logger.info(f"MAE = {mean_absolute_error}")

        avg_score = np.mean(error_scores)
        return avg_score

    logger.info("Beginning hyperparameter search")
    sampler = TPESampler(seed=69)
    study = optuna.create_study(study_name="optuna_study", direction="minimize", sampler=sampler)
    study.optimize(func=objective, n_trials=hyperparameter_trials)

    # Get the dictionary of the best hyperparameters and the error that they produce
    best_hyperparams = study.best_params
    best_value = study.best_value

    logger.info(f"The best hyperparameters for {model_fn} are:")
    for key, value in best_hyperparams.items():
        logger.info(f"{key}:{value}")

    logger.info(f"Best MAE: {best_value}")
    experiment = Experiment(
        api_key=settings.comet_api_key,
        workspace=settings.comet_workspace,
        project_name=settings.comet_project_name
    )

    experiment.set_name(f"Hyperparameter Tuning of {model_name} model")
    experiment.log_metric(name="Cross validation M.A.E", value=best_value)
    experiment.add_tags(tags=[scenario, model_name])
    experiment.end()

    return best_hyperparams
