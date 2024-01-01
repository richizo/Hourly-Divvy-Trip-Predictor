import random
import pickle
import pathlib
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.insert(0, "/home/kobina/cyclistic-bike-sharing-data/src")


# MEMORY MANAGEMENT

def view_memory_usage(
        data: pd.DataFrame,
        column: str
):
    """This function allows us to view the amount of memory being
    used by one or more columns of a given dataframe."""

    yield data[column].memory_usage(index=False, deep=True)


def change_column_data_type(
        data: pd.DataFrame,
        columns: list,
        to_format: str
):
    """
    This function changes the datatype of one or more columns of 
    a given dataframe.
    """

    data[columns] = data[columns].astype(to_format)


# ROUNDING COORDINATES
def add_rounded_coordinates_to_dataframe(
        data: pd.DataFrame,
        decimal_places: int,
        start_or_stop: str
):
    """This function takes the latitude and longitude columns of a dataframe,
    rounds them down to a specified number of decimal places, and creates
    a new column for these."""

    new_lats = []
    new_longs = []

    for latitude in tqdm(data[f"{start_or_stop}_latitude"].values):
        new_lats.append(
            np.round(latitude, decimals=decimal_places)
        )

    for longitude in tqdm(data[f"{start_or_stop}_longitude"].values):
        new_longs.append(
            np.round(longitude, decimals=decimal_places)
        )

    # We put these lists in the dataframe. I have done so using Pandas' insert function to avoid the SettingCopy Warning
    data.insert(
        loc=data.shape[1],
        column=f"rounded_{start_or_stop}_latitude",
        value=pd.Series(new_lats),
        allow_duplicates=False
    )

    data.insert(
        loc=data.shape[1],
        column=f"rounded_{start_or_stop}_longitude",
        value=pd.Series(new_longs),
        allow_duplicates=False
    )


def add_column_of_rounded_points(
        data: pd.DataFrame,
        start_or_stop: str
):
    """Make a column which consists of points containing the rounded latitudes and longitudes."""

    points = [
        point for point in zip(data[f"rounded_{start_or_stop}_latitude"], data[f"rounded_{start_or_stop}_longitude"])
    ]

    data.insert(
        loc=data.shape[1],
        column=f"rounded_{start_or_stop}_points",
        value=pd.Series(points),
        allow_duplicates=False)


def make_new_station_ids(
        data: pd.DataFrame,
        start_or_stop: str
) -> dict:
    """
    This function makes a list of random numbers for each unique point, and 
    associates each point with a corresponding number. This effectively creates new 
    IDs for each location.
    """

    num_unique_points = len(data[f"rounded_{start_or_stop}_points"].unique())

    # Make a list of k values consisting of values taken from the population
    randoms = random.sample(population=range(num_unique_points), k=num_unique_points)

    points_and_new_ids = {}
    for point, value in tqdm(zip(data[f"rounded_{start_or_stop}_points"].unique(), randoms)):
        points_and_new_ids[point] = value

    return points_and_new_ids


def save_dict(
        dictionary: dict,
        folder: pathlib.PosixPath,
        file_name: str
):
    """ Save a dictionary (as a .pkl file) into a specified folder,
    and with a specified file name"""

    with open(f"{folder}/{file_name}", "wb") as temp:
        pickle.dump(dictionary, temp)


# Form a column of said IDs (in the appropriate order)
def add_column_of_ids(
        data: pd.DataFrame,
        start_or_stop: str,
        points_and_ids: dict
):
    """Take each point, and the ID which corresponds to it
    (within its dictionary), and put those IDs in the relevant dataframe 
    (in a manner that matches each point with its ID row-wise)."""

    location_ids = [
        points_and_ids[point] for point in list(data.loc[:, f"rounded_{start_or_stop}_points"]) if
        point in points_and_ids.keys()
    ]

    data.insert(
        loc=data.shape[1],
        column=f"{start_or_stop}_station_id",
        value=pd.Series(location_ids),
        allow_duplicates=False
    )
