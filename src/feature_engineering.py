import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

def average_trips_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    
    X.insert(
        loc=X.shape[1], 
        column="average_trips_last_4_weeks",
        value= 0.25*(X[f"trips_previous_{7*24}_hour"] + \
                     X[f"trips_previous_{2*7*24}_hour"] + \
                     X[f"trips_previous_{3*7*24}_hour"] + \
                     X[f"trips_previous_{4*7*24}_hour"])
    )
   
    return X


class TemporalFeatureEngineeringStarts(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        
        X.insert(
            loc=X.shape[1],
            column="hour",
            value=X["start_hour"].dt.hour
        )

        X.insert(
            loc=X.shape[1],
            column="day_of_the_week",
            value=X["start_hour"].dt.dayofweek
        )

        return X.drop("start_hour", axis = 1)


class TemporalFeatureEngineeringStops(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):

        X.insert(
            loc=X.shape[1],
            column="hour",
            value=X["stop_hour"].dt.hour
        )

        X.insert(
            loc=X.shape[1],
            column="day_of_the_week",
            value=X["stop_hour"].dt.dayofweek
        )

        return X.drop("stop_hour", axis = 1)


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
