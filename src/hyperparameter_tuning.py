import tqdm 
import optuna 
import numpy as np
import pandas as pd 

from typing import Optional, Callable, Dict

from sklearn.linear_model import Lasso 
import lightgbm as LGBMRegressor
from xgboost import XGBRegressor, XGBRFRegressor

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.logger import get_logger
from src.feature_engineering import get_start_pipeline, get_stop_pipeline

from comet_ml import Experiment


def sampled_hyperparams(
    model_fn: Callable,
    trial: optuna.trial.Trial) -> Dict[str, Union[str, int, float]]:
  
  """
  Return the values to be considered 

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
      "num_leaves": trial.suggest_int("num_leaves",8, 64),
      "max_depth": trial.suggest_int("max_depth", 3, 10),
      "subsample": trial.suggest_int("subsample", 0.1, 1),
      "n_estimators": trial.suggest_int("n_estimators", 20, 150),
      "learning_rate": trial.suggest_float("learning_rate", 0.1, 1),
      "importance_type": trial.suggest_categorical("importance_type", ["split", "gain"]),
      "colsample_by_tree": trial.suggest_float("colsample_by_tree", 0.1, 1),
      "colsample_by_level": trial.suggest_float("colsample_by_tree", 0.1, 1),
      "colsample_by_node": trial.suggest_float("colsample_by_node", 0.1, 1)
    }
  
  elif model_fm == xgboost:
    
    return {
      "objective": "reg:absolute_error",
      "eta": trial.suggest_float("eta", 0,1, 1),
      "max_depth": trial.suggest_int("max_depth", 3, 10),
      "alpha": trial.suggest_float("alpha", 0, 2),
      "subsample": trial.suggest_int("subsample", 0.1, 1),
      "colsample_by_tree": trial.suggest_float("colsample_by_tree", 0.1, 1),
      "colsample_by_level": trial.suggest_float("colsample_by_tree", 0.1, 1),
      "colsample_by_node": trial.suggest_float("colsample_by_node", 0.1, 1)      
    }
  
  else:
    
    raise NotImplementedError("This model has not been implemented.")
    
  
    
def optimise_hyperparams(
  model_fn: Callable,
  hyperparem_trials: int,
  scenario: str,
  X: pd.Dataframe,
  y: pd.Series,
  experiment = Experiment
) -> Dict:
  
  logger = get_logger()
  assert model_fn in [Lasso, LGBMRegressor, XGBRegressor]
  
  def objective(trial: optuna.trial.Trial) -> float:
    
    """ The error function we want to optimise. """
    
    hyperparams = sampled_hyperparams(model_fn=model_fn, trial=trial)
  
    
    # Cross-validation
    scores = []
    tss = TimeSeriesSplit(n_splits=5)
    logger.info(f"{trial.number=}") 
    
    # Use TSS to split the features and target variables for training and validation
    for split_number, (train_indices, val_indices) in tqdm(enumerate(tss.split(X))):
      
      X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
      y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
      
      logger.info(f"{split_number=}")
      logger.info(f"{len(X_train)}=")
      logger.info(f"{len(X_val)}=")
      
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
      
      pipeline.fit(X_train, y_train)
      
      y_pred = pipeline.predict(X_val)
      error = mean_absolute_error(y_val, y_pred)
      scores.append(error)
      
      logger.info(f"{error=}")
      
    avg_score = np.mean(scores)
    
    return avg_score
#LGBMRegressor.callback.reset_parameter