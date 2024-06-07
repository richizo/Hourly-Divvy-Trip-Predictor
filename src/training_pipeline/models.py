import pandas as pd 
import numpy as np 

from pathlib import Path 

from loguru import logger
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from src.setup.paths import TRAINING_DATA
from src.setup.config import settings
from src.feature_pipeline.preprocessing import DataProcessor
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparams 
from src.feature_pipeline.feature_engineering import FeatureEngineering


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


def make_training_data() -> pd.DataFrame:
    training_sets = DataProcessor(year=settings.year)._make_training_data()
    training_data = training_sets[0] if scenario == "start" else training_sets[1]
    return training_data


def build_model(model_fn: BaseModel|Lasso|LGBMRegressor|XGBRegressor, scenario: str, tune_hyperparams: bool) -> None:
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
    models_and_tags = {Lasso: "Lasso", LGBMRegressor: "LGBMRegressor", XGBRegressor: "XGBRegressor"}
    data_path = Path(TRAINING_DATA/f"{scenario}s.parquet")
        
    if data_path.is_file():
        logger.success("The training data has already been created")
        training_data = pd.read_parquet(path=data_path)

    else:
        logger.info("No training data has been stored. Creating the dataset will take a long time...")
        training_data = make_training_data()

        logger.info("Training data created successfully")
        target = training_data["trips_next_hour"] 
        features = training_data.drop("trips_next_hour", axis=1)

        if tune_hyperparams:
            logger.info(
                f"Tuning hyperparameters of the {models_and_tags[model_fn]} model. Have a snack and watch some One Piece"
            )

            best_hyperparams = optimise_hyperparams(
                model_fn=model_fn, 
                hyperparam_trials = 5, 
                scenario = scenario, 
                X = features, 
                y = target
            )

            model = make_pipeline(
                get_start_pipeline() if scenario == "start" else get_stop_pipeline(),
                model_fn(**best_hyperparams)
            )  

        else:
            model = make_pipeline(
                get_start_pipeline() if scenario == "start" else get_stop_pipeline(),
                model_fn()
            )  

        logger.info("Saving the model so we won't have to go through that again...")
        pickle_name = f"Best {models_and_tags[model_fn]} model for {scenario}s.pickle"
        with open(MODELS_DIR/pickle_name, "wb") as place:
            pickle.dump(obj=model, file=place)
