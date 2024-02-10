import os 
import optuna 
import numpy as np
import pandas as pd 

from comet_ml import Experiment
from typing import Optional, Callable, Dict, Union

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso 

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.logger import get_logger
from src.feature_engineering import get_start_pipeline, get_stop_pipeline


def sampled_hyperparams(
    model_fn: Callable,
    trial: optuna.trial.Trial
    ) -> Dict[str, Union[str, int, float]]:
  
  """
  Return the range of values of each hyperparameter under consideration.

  Returns:
      dict: the range of hyperparemeter values to be considered
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
      "subsample": trial.suggest_int("subsample", 0.1, 1),
      "colsample_by_tree": trial.suggest_float("colsample_by_tree", 0.1, 1),
      "colsample_by_level": trial.suggest_float("colsample_by_level", 0.1, 1),
      "colsample_by_node": trial.suggest_float("colsample_by_node", 0.1, 1)      
    }
  
  else:
    
    raise NotImplementedError("This model has not been implemented")
    
    
def optimise_hyperparams(
  model_fn: Callable,
  hyperparam_trials: int,
  scenario: str,
  X: pd.DataFrame,
  y: pd.Series
) -> Dict:
  
  logger = get_logger()
  assert model_fn in [Lasso, LGBMRegressor, XGBRegressor]
  
  def objective(trial: optuna.trial.Trial) -> float:
    
    """
    Perform Time series cross validation, fit a pipeline to it (equipped with the)
    selected model, and return the average error across all cross validation splits.
    
    Args:

    model_fn: the model architecture to be used
    hyperparam_trials: the number of optuna trials that will be run per model
    scenario: whether we are looking at the trip start or trip stop data.
    
    X: the dataframe of features
    y: the pandas series which contains the target variable
    experiment: the defined CometML experiment object
    
    Returns:
        float: Average error per split.
    """
    
    hyperparams = sampled_hyperparams(model_fn=model_fn, trial=trial)
  
    scores = []
    tss = TimeSeriesSplit(n_splits=5)
    logger.info(f"Start Trial {trial.number}") 
    
    # Use TSS to split the features and target variables for training and validation
    for split_number, (train_indices, val_indices) in enumerate(tss.split(X)):
      
      logger.info(f"Performing split number {split_number}")
     
      if scenario == "start" and "start_station_id" in X.columns:
        
        pipeline = make_pipeline(
          get_start_pipeline(),
          model_fn(**hyperparams)
        )
        
      if scenario == "stop" and "stop_station_id" in X.columns:
        
        pipeline = make_pipeline(
          get_stop_pipeline(),
          model_fn(**hyperparams)
        )
      
      pipeline.fit(
        X.iloc[train_indices], y.iloc[train_indices]
      )
      
      y_pred = pipeline.predict(X.iloc[val_indices])
      error = mean_absolute_error(y.iloc[val_indices], y_pred)
      scores.append(error)
      
    avg_score = np.mean(scores)
    
    return avg_score  
  
  logger.info("Beginning hyperparameter search")
  
  study = optuna.create_study(direction="minimize")
  study.optimize(func=objective, n_trials=hyperparam_trials)
  
  # Get the dictionary of the best hyperparameters and the error that they produce
  best_hyperparams = study.best_params
  best_value = study.best_value
  
  logger.info(f"The best hyperparameters for {model_fn} are:")
  
  for key, value in best_hyperparams.items():
    logger.info(f"{key}:{value}")
    
  logger.info(f"Best MAE: {best_value}")
  
  # Log an experiment with CometML  
  experiment = Experiment(
    api_key=os.environ["COMET_API_KEY"],
    workspace=os.environ["COMET_WORKSPACE"],
    project_name=os.environ["COMET_PROJECT_NAME"]
  )

  experiment.log_metric(name="Cross validation M.A.E", value=best_value)
  
  #Attach a tag to a CometML experiment to show which dataset was  worked on, and 
  # include another tag indicating the model type.
  experiment.add_tag(scenario)
  
  models_and_tags = {
    Lasso: "Lasso", LGBMRegressor: "LGBMRegressor", XGBRegressor: "XGBRegressor"
  }

  experiment.add_tag(
    models_and_tags[model_fn]
  )
  
  experiment.end()
  
  return best_hyperparams
  