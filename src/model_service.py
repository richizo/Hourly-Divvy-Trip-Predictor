from pathlib import Path

from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import pickle
from typing import Type, Callable
from loguru import logger  

from src.model import build_model
from src.paths import MODELS_DIR


models_and_tags = {
    Lasso: "Lasso", 
    LGBMRegressor: "LGBMRegressor", 
    XGBRegressor: "XGBRegressor"
  }

class ModelService:
  
  def __init__(self):
    self.model =  None
    
    
  def load_model(
    self, 
    model_fn: Callable, 
    scenario: str
    ) -> None:
  
    """
    Loads the requested model if it already exists, or builds it if 
    the model isn't already saved in the file system.
    
    Args:
      model_fn -- the class of the model that is desired
      scenario -- a string indicating whether we are training on
                start or stop data. The only accepted answers are
                "start" and "stop"
    """
    
    # Check whether the model is already saved
    pickle_name = f"Best {models_and_tags[model_fn]} model.pickle"

    model_path = Path(MODELS_DIR/pickle_name)
  
    # If it is, load it.
    if not model_path.exists():
      
      logger.info("The requested model hasn't been trained yet -> Looking for training data")
      build_model(model_fn=model_fn, scenario=scenario)
        
    with open(model_path, "rb") as place:
        
      self.model = pickle.load(file=place)  
    
    
  def predict(self, inputs):
    
    return self.model.predict([inputs])
  