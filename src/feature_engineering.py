import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

from warnings import simplefilter


def average_trips_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    
    if "average_trips_last_4_weeks" in X.columns:
        
        pass
    
    else:
        
        simplefilter(
            action = "ignore",
            category=pd.errors.PerformanceWarning
        )
    
        X.insert(
            loc=X.shape[1], 
            column="average_trips_last_4_weeks",
            value=0.25*(
                X[f"trips_previous_{1*7*24}_hour"] + X[f"trips_previous_{2*7*24}_hour"] + \
                X[f"trips_previous_{3*7*24}_hour"] + X[f"trips_previous_{4*7*24}_hour"]
                        )
        )
   
    return X


class TemporalFeatureEngineeringStarts(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, scenario: str = "start", y=None):
        return self 

    def transform(self, X: pd.DataFrame, scenario: str = "start", y=None):
        
        X.insert(
            loc=X.shape[1],
            column="hour",
            value=X[f"{scenario}_hour"].dt.hour
        )   
        
        
        X.insert(
            loc=X.shape[1],
            column="day_of_the_week",
            value=X[f"{scenario}_hour"].dt.dayofweek
        )   
                
        return X.drop(f"{scenario}_hour", axis = 1)


class TemporalFeatureEngineeringStops(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        
        return self 

    def transform(self, X: pd.DataFrame, scenario: str = "stop", y=None):
        
        X.insert(
            loc=X.shape[1],
            column="hour",
            value=X[f"{scenario}_hour"].dt.hour
        )   
        
        
        X.insert(
            loc=X.shape[1],
            column="day_of_the_week",
            value=X[f"{scenario}_hour"].dt.dayofweek
        )   
                
        return X.drop(f"{scenario}_hour", axis = 1)
        
        
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline


def get_start_pipeline() -> Pipeline:
    
    return make_pipeline(
        FunctionTransformer(func = average_trips_last_4_weeks, validate = False),
        TemporalFeatureEngineeringStarts()
    )


def get_stop_pipeline() -> Pipeline:
    
    return make_pipeline(
        FunctionTransformer(func = average_trips_last_4_weeks, validate = False),
        TemporalFeatureEngineeringStops()
    )
