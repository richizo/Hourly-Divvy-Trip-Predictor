import optuna 
import numpy as np
import pandas as pd 

from comet_ml import Experiment
from loguru import logger 
from optuna.samplers import TPESampler

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso 

from src.setup.config import settings
from src.training_pipeline.models import BaseModel


def sampled_hyperparams(
    model_fn: BaseModel|Lasso|LGBMRegressor|XGBRegressor,
    trial: optuna.trial.Trial
  ) -> dict[str, str|int|float]:
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
      "subsample": trial.suggest_int("subsample", 0.1, 1)
    }
  
  else:
    raise NotImplementedError("This model has not been implemented")
    
    
def optimise_hyperparams(
  model_fn: BaseModel|Lasso|LGBMRegressor|XGBRegressor,
  hyperparam_trials: int,
  scenario: str,
  X: pd.DataFrame,
  y: pd.Series
  ) -> dict:
  
  assert model_fn in [Lasso, LGBMRegressor, XGBRegressor]
  
  models_and_tags = {
    Lasso: "Lasso", 
    LGBMRegressor: "LGBMRegressor", 
    XGBRegressor: "XGBRegressor"
  }
  
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
    logger.info(f"Starting Trial {trial.number}") 
    
    # Use TSS to split the features and target variables for training and validation
    for split_number, (train_indices, val_indices) in enumerate(tss.split(X)):
      
      logger.info(f"Performing split number {split_number}")
  
      model_fn.fit(
        X.iloc[train_indices], y.iloc[train_indices]
      )
     
      y_pred = model_fn.predict(X.iloc[val_indices])
      error = mean_absolute_error(y.iloc[val_indices], y_pred)
      scores.append(error)
        
    avg_score = np.mean(scores)
    
    return avg_score  
   
  logger.info("Beginning hyperparameter search")
  
  sampler = TPESampler(seed=69)
  
  study = optuna.create_study(
    study_name="optuna_study",
    direction="minimize", 
    sampler=sampler
    )
  
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
    api_key=settings.comet_api_key,
    workspace=settings.comet_workspace,
    project_name=settings.comet_project_name
  )
  
  # Set the name of the experiment and log the target metric.
  experiment.set_name(f"Hyperparameter Tuning of {models_and_tags[model_fn]} model")
  experiment.log_metric(name="Cross validation M.A.E", value=best_value)
  
  # Attach a tags to the CometML experiment to show whether we are training on start 
  # or stop data, and to indicate the model type.
  experiment.add_tags(
    tags=[scenario, models_and_tags[model_fn]]
  )
    
  experiment.end()
 
  return best_hyperparams
  