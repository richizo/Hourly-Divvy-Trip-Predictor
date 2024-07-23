# Utilities
import json
import random
import pathlib
from tqdm import tqdm

# Config
from src.setup.config import config

# Data Manipulation and Access
import numpy as np
import pandas as pd

# Reverse Geocoding
from geopy import Photon

# Custom modules
from src.setup.paths import GEOGRAPHICAL_DATA


class ReverseGeocoder:
    def __init__(self, scenario: str, geodata: pd.DataFrame, simple: bool = True) -> None:
        self.geodata = geodata
        self.simple = simple
        self.scenario = scenario
        self.coordinates = geodata["coordinates"].values
        self.station_ids = geodata[f"{scenario}_station_id"].unique()

    def find_points_simply(self) -> dict[int, list[float]]:

        simple_match = {}
        number_of_rows = self.geodata.shape[0]

        for code in tqdm(range(number_of_rows)):
            for station_id in self.station_ids:
                if station_id not in simple_match.keys() and station_id == self.geodata.iloc[code, 0]:
                    simple_match[station_id] = self.geodata.iloc[code, 1]

        return simple_match

    def reverse_geocode(self, save: bool = True) -> dict[str, list[float]]:
        """
        Perform reverse geocoding of each coordinate in the dataframe (avoiding duplicates), and make 
        a dictionary of coordinates and their station addresses. That dictionary can then be saved, and 
        is returned.

        Returns:
            dict[str, list[float]]: the station IDs obtained from reverse geocoding, and the original
                                    coordinates.
        """
        addresses_and_points = {}
        geocoder = Photon(user_agent=config.email)
        
        coordinate_source = self.find_points_simply().values() if self.simple else self.coordinates
        coordinates = tqdm(iterable=coordinate_source, desc="Reverse geocoding the coordinates")

        for coordinate in coordinates:
            if coordinate in addresses_and_points.values():
                addresses_and_points[str(geocoder.reverse(query=coordinate, timeout=120))] = coordinate
        if save:    
            with open(GEOGRAPHICAL_DATA/f"{self.scenario}_station_names_and_coordinates.json", mode="w") as file:
                json.dump(addresses_and_points, file)

        return addresses_and_points

    def put_station_names_in_geodata(self, station_names_and_coordinates: dict) -> pd.DataFrame:
        
        station_names_to_add = []

        for coordinate in self.geodata["coordinates"]:  
            if coordinate in station_names_and_coordinates.values():
                station_names_to_add.append(station_names_and_coordinates[coordinate])

        return pd.concat(
                [self.geodata, pd.Series(data=station_names_to_add)]
            )

class RoundingCoordinates:

    def __init__(self, scenario: str, data: pd.DataFrame, decimal_places: int) -> None:
        """
        The contents of this class are only to be used if the data being processed is so voluminous that it 
        poses a such a problem (in terms of memory and time) that the compromise of geographical accuracy 
        resulting from its use can be justified.

        Args:
            data (pd.DataFrame): the data being processed

            decimal_places (int): the number of decimal places to which we will round the coordinates. 
                                The original coordinates are written in 6 decimal places. For each 
                                decimal place that is lost, the accuracy of the coordinates degrades 
                                by a factor of 10 meters

            scenario (str): whether we are looking at "start" (departures) or "end" (arrival) data.
        """
        self.data = data 
        self.scenario = scenario
        self.decimal_places = decimal_places

    def add_rounded_coordinates_to_dataframe(self) -> None:
        """
        This function takes the latitude and longitude columns of a dataframe,
        rounds them down to a specified number of decimal places, and creates
        a new column for these.

        """
        new_latitudes = []
        new_longitudes = []

        latitudes = tqdm(
            iterable=self.data[f"{self.scenario}_lat"].values,
            desc="Working on latitudes"
        )

        longitudes = tqdm(
            iterable=self.data[f"{self.scenario}_lng"].values,
            desc="Working on longitudes"
        )

        for latitude in latitudes:
            new_latitudes.append(
                np.round(latitude, decimals=self.decimal_places)
            )

        for longitude in longitudes:
            new_longitudes.append(
                np.round(longitude, decimals=self.decimal_places)
            )

        # Insert the rounded latitudes into the dataframe
        self.data.insert(
            loc=self.data.shape[1],
            column=f"rounded_{self.scenario}_lat",
            value=pd.Series(new_latitudes),
            allow_duplicates=False
        )

        # Insert the rounded longitudes into the dataframe
        self.data.insert(
            loc=self.data.shape[1],
            column=f"rounded_{self.scenario}_lng",
            value=pd.Series(new_longitudes),
            allow_duplicates=False
        )

    def add_column_of_rounded_points(self):
        """
        Make a column which consists of points containing the rounded latitudes and longitudes.
        """
        points = list(
            zip(
                self.data[f"rounded_{self.scenario}_lat"], self.data[f"rounded_{self.scenario}_lng"]
            )
        )

        self.data.insert(
            loc=self.data.shape[1],
            column=f"rounded_{self.scenario}_points",
            value=pd.Series(points),
            allow_duplicates=False
        )

    def make_station_ids_from_unique_coordinates(self) -> dict[float, int]:
        """
        This function makes a list of random numbers for each unique point, and 
        associates each point with a corresponding number. This effectively creates new 
        IDs for each location.
        """
        unique_coordinates = self.data[f"rounded_{self.scenario}_points"].unique()
        num_unique_points = len(unique_coordinates)

        # Set a seed to ensure reproducibility. 
        random.seed(69)

        # Make a random mixture of the numbers from 0 to len(num_unique_points) 
        station_ids = random.sample(population=range(num_unique_points), k=num_unique_points)

        # Make a dictionary of points
        points_and_new_ids = {}

        for point, value in tqdm(zip(unique_coordinates, station_ids)):
            points_and_new_ids[point] = value

        return points_and_new_ids

    def add_column_of_ids(self, points_and_ids: dict) -> None:
        """
        Take each point, and the ID which corresponds to it (within its dictionary),
        and put those IDs in the relevant dataframe (in a manner that matches each 
        point with its ID row-wise).

        Args:
            points_and_ids (dict): dictionary of unique coordinates and IDs.
        """
        station_ids = [
            points_and_ids[point] for point in list(self.data.loc[:, f"rounded_{self.scenario}_points"]) if
            point in points_and_ids.keys()
        ]

        self.data.insert(
            loc=self.data.shape[1],
            column=f"{self.scenario}_station_id",
            value=pd.Series(station_ids),
            allow_duplicates=False
        )

    @staticmethod
    def save_geodata_dict(points_and_ids: dict, folder: pathlib.PosixPath, file_name: str):
        """
        Save the geographical data which consists of the station IDs and their corresponding
        coordinates as a geojson file. It was necessary to swap the keys and values (the coordinates
        and IDs respectively) because json.dump() does not allow tuples to be keys.

        Args:
            points_and_ids (dict): the dictionary of coordinates and their (new) IDs.

            folder (pathlib.PosixPath): the directory where the file is to be saved

            file_name (str): the name of the .pkl file
        """
        swapped_dict = {
            station_id: point for point, station_id in points_and_ids.items()
        }
        with open(f"{folder}/{file_name}.geojson", mode="w") as file:
            json.dump(swapped_dict, file)


def view_memory_usage(data: pd.DataFrame, column: str) -> pd.Series:
    """
    This function allows us to view the amount of memory being
    used by one or more columns of a given dataframe.
    """
    yield data[column].memory_usage(index=False, deep=True)


def change_column_data_type(data: pd.DataFrame, columns: list, to_format: str):
    """
    This function changes the datatype of one or more columns of 
    a given dataframe.
    """
    data[columns] = data[columns].astype(to_format)
