import numpy as np 
import pandas as pd 

from datetime import datetime

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

from src.setup.paths import TRAINING_DATA


class BaseModel():

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.data = pd.read_parquet(TRAINING_DATA/f"{scenario}s.parquet")
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        pass 

    def train_test_split(self, cutoff_date: datetime) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        This is just a primitive splitting function that treats all data before a certain date
        as training data, and everything after thatas test data

        Args:
            cutoff_data: 
            target_column:

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: the training and test sets 
                                                                     (features and targets)
        """
        training_data = self.data[self.data[f"{self.scenario}_hour"] < cutoff_date].reset_index(drop=True)
        test_data = self.data[self.data[f"{self.scenario}_hour"] > cutoff_date].reset_index(drop=True)

        x_train = training_data.drop(columns=["trips_next_hour"])
        y_train = training_data["trips_next_hour"]

        x_test = test_data.drop(columns=["trips_next_hour"])
        y_test = test_data["trips_next_hour"]

        return x_train, y_train, x_test, y_test

    def predict(self, x_test: pd.DataFrame) -> np.array:
        return x_test["trips_previous_1_hour"]

    def compute_error(self, y_true: float, y_pred: float):
        return mean_absolute_error(y_true=y_true)


def get_model(model_name: str) -> BaseModel|XGBRegressor|LGBMRegressor:
    """
    
    Args:
        model_name (str): Capitalisations of the following names are 
                          allowoed:
                          'base' for the base model, 
                          'xgboost' for XGBRegressor,
                          'lightgbm' for LGBMRegressor, 
                          'lasso' for Lasso.

    Returns:
        BaseModel|XGBRegressor|LGBMRegressor: the requested model
    """
    models_and_names = {
        "lasso": Lasso,
        "lightgbm": LGBMRegressor,
        "xgboost": XGBRegressor
    }
    if model_name.lower() in models_and_names.keys():
        return models_and_names[model_name.lower()]
