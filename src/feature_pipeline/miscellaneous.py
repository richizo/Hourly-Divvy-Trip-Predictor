"""
Consists of :
- functions that are not in use at the moment, but may prove useful in the future
- functions that are needed in aspects of the code that are yet to be fully fleshed out
"""

import json
import random
import pathlib

import numpy as np
import pandas as pd

from tqdm import tqdm


def view_memory_usage(data: pd.DataFrame, column: str) -> pd.Series:
    """
    This function allows us to view the amount of memory being used by one or more columns of
    a given dataframe.
    """
    yield data[column].memory_usage(index=False, deep=True)


def save_geodata_dict(dictionary: dict, folder: pathlib.PosixPath, file_name: str):
    """
    Save the geographical data which consists of the station IDs and their corresponding
    coordinates as a geojson file. It was necessary to swap the keys and values (the coordinates
    and IDs respectively) because json.dump() does not allow tuples to be keys.

    Args:
        dictionary (dict): the target dictionary
        folder (pathlib.PosixPath): the directory where the file is to be saved
        file_name (str): the name of the .pkl file
    """
    swapped_dict = {station_id: point for point, station_id in dictionary.items()}
    with open(f"{folder}/{file_name}.geojson", "w") as file:
        json.dump(swapped_dict, file)


def make_ids_for_each_coordinate(data: pd.DataFrame, scenario: str) -> dict:
    """
    This function makes a list of random numbers for each unique coordinate, and
    associates each coordinate with a corresponding number.
    """
    num_unique_points = len(
        data[f"rounded_{scenario}_points"].unique()
    )

    # Set a seed to ensure reproducibility. 
    random.seed(69)

    # Make a list of k values consisting of values taken from the population
    station_ids = random.sample(population=range(num_unique_points), k=num_unique_points)

    # Make a dictionary of points
    points_and_new_ids = {}

    for point, value in tqdm(zip(data[f"rounded_{scenario}_points"].unique(), station_ids)):
        points_and_new_ids[point] = value

    return points_and_new_ids


# Form a column of said IDs (in the appropriate order)
def add_column_of_ids(data: pd.DataFrame, scenario: str, points_and_ids: dict) -> pd.DataFrame:
    """
    Take each point, and the ID which corresponds to it (based on the provided dictionary),
    and put those IDs in the relevant dataframe (in a manner that matches each
    point with its ID row-wise).

    Args:
        data:
        scenario:
        points_and_ids:

    Returns:

    """
    data[f"{scenario}_station_id"] = data[f"rounded_{scenario}_points"].map(points_and_ids)
    return data
