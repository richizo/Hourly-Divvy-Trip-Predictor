import optuna 
import pandas as pd 

import lightgbm as LGBMRegressor
from typing import Optional, Callable, Dict

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering import get_start_pipeline, get_stop_pipeline


def sample_hyperparams(
    model_fn: Callable,
    trial: optuna.trial.Trial) -> Dict[str, Union[str, int, float]]:
  
  if model_fn == Lasso:
    
    return {
      "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True)
    }
    
  elif model_fn == LGBMRegressor:
    
    return {
      "metric": "mae",
      "verbose": -1, 
      "num_leaves": trial.suggest_int("num_leaves", 2, 300),
      "max_depth": trial.suggest_int("max_depth", 5, 10),
      "n_estimators": trial.suggest_int("n_estimators", 20, 150)
      
    }
    