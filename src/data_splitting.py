from datetime import datetime
import pandas as pd


def train_test_split(
        data: pd.DataFrame,
        scenario: str,
        cutoff_date: datetime,
        target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    This is just a primitive splitting function that treats all data
    before a certain date as training data, and everything after that
    as test data
    """

    training_data = data[data[f"{scenario}_hour"] < cutoff_date].reset_index(drop=True)

    test_data = data[data[f"{scenario}_hour"] > cutoff_date].reset_index(drop=True)

    x_train = training_data.drop(columns=[target_column])
    y_train = training_data[target_column]

    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return x_train, y_train, x_test, y_test
