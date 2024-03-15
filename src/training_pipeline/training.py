import pandas as pd 
from pathlib import Path 
import pickle 
from typing import Callable

from requests import Request
from optuna.trial import FrozenTrial
from flask import Flask, jsonify 
from loguru import logger
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


from src.feature_pipeline.preprocessing import make_training_data
from src.feature_pipeline.feature_engineering import average_trips_last_4_weeks, get_start_pipeline, get_stop_pipeline
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparams
from src.setup.paths import MODELS_DIR, TRAINING_DATA


def build_model(
  model_fn: Callable, 
  scenario: str
  ) -> None:
  
  """
  Fetches or builds a tuned version of the model of choice. 
  
  The function first checks for the existence of the training data, 
  and builds it if it doesn't find the file. Then it checks for a
  saved model. If it doesn't find a model, it will go on to build 
  one, tune its hyperparameters, save the resulting model.
  
  Args: 
    model_fn -- the class of the model that is desired
    scenario -- a string indicating whether we are training on
                start or stop data. The only accepted answers are
                "start" and "stop"
  """
  
  models_and_tags = {
    Lasso: "Lasso", 
    LGBMRegressor: "LGBMRegressor", 
    XGBRegressor: "XGBRegressor"
  }
  
  data_path = Path(TRAINING_DATA/f"{scenario}s.parquet")
        
  if data_path.is_file():

    logger.info("The training data has already been created")
    training_data = pd.read_parquet(path=data_path)

  else:
    
    logger.info("No training data has been stored. Creating the dataset will take a long time.")
    training_data = make_training_data(scenario=scenario)

  logger.info("Training data created successfully")
  
  target = training_data["trips_next_hour"] 
  features = training_data.drop("trips_next_hour", axis=1)

  logger.info(f"Tuning hyperparameters of the {models_and_tags[model_fn]} model. Have a coffee")
  
  best_hyperparams = optimise_hyperparams(
      model_fn=LGBMRegressor, 
      hyperparam_trials = 5, 
      scenario = scenario, 
      X = features, 
      y = target
  )

  if scenario == "start":
    
    model = make_pipeline(
      get_start_pipeline(),
      model_fn(**best_hyperparams)    
    )

  if scenario == "stop":
    
    model = make_pipeline(
      get_stop_pipeline(),
      model_fn(**best_hyperparams)    
    )
    
  logger.info("Saving the model so we won't have to go through that again.")

  pickle_name = f"Best {models_and_tags[model_fn]} model for {scenario}s.pickle"

  with open(MODELS_DIR/pickle_name, "wb") as place:
    
    pickle.dump(obj=model, file=place)
  