# Utilities
import json
import random

from tqdm import tqdm
from loguru import logger 
from pathlib import Path, PosixPath

# Config
from src.setup.config import config

# Data Manipulation and Access
import numpy as np
import pandas as pd

# Geocoding & Reverse Geocoding
from geopy import Photon

# Custom modules
from src.setup.paths import INDEXER_TWO


class RoundingCoordinates:
    """
    The contents of this class are only to be used if the data being processed is so voluminous that it 
    poses a such a problem (in terms of memory and time) that the compromise of geographical accuracy 
    resulting from its use can be justified.
    """
    def __init__(self, scenario: str, data: pd.DataFrame, decimal_places: int | None) -> None:
        """
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

    def add_column_of_rounded_coordinates_to_dataframe(self) -> None:
        """
        This function takes the latitude and longitude columns of a dataframe, rounds them down to a 
        specified number of decimal places, and makes a column which consists of points containing the 
        rounded latitudes and longitudes.
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

        for coordinate_name, coordinate in zip(
            ["lat", "lng"], [new_latitudes, new_longitudes]
        ):
            self.data.insert(
                loc=self.data.shape[1],
                column=f"rounded_{self.scenario}_{coordinate_name}",
                value=pd.Series(coordinate),
                allow_duplicates=False
            )

        # Remove the original latitudes and longitudes
        cleaned_data = cleaned_data.drop(
            columns=[f"{start_or_end}_lat", f"{start_or_end}_lng"]
        )

        rounded_points = list(
            zip(
                self.data[f"rounded_{self.scenario}_lat"], self.data[f"rounded_{self.scenario}_lng"]
            )
        )

        # Insert the rounded() coordinates as points
        self.data.insert(
            loc=self.data.shape[1],
            column=f"rounded_{self.scenario}_points",
            value=pd.Series(rounded_points),
            allow_duplicates=False
        )

        # Remove the rounded latitudes and longitudes that we added
        cleaned_data.drop(
            columns=[f"rounded_{start_or_end}_lat", f"rounded_{start_or_end}_lng"],
            inplace=True
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
    def save_geodata_dict(points_and_ids: dict, folder: PosixPath, file_name: str):
        """
        Save the geographical data which consists of the station IDs and their corresponding
        coordinates as a geojson file. It was necessary to swap the keys and values (the coordinates
        and IDs respectively) because json.dump() does not allow tuples to be keys.

        Args:
            points_and_ids (dict): the dictionary of coordinates and their (new) IDs.

            folder (PosixPath): the directory where the file is to be saved

            file_name (str): the name of the .pkl file
        """
        swapped_dict = {
            station_id: point for point, station_id in points_and_ids.items()
        }

        with open(f"{folder}/{file_name}.geojson", mode="w") as file:
            json.dump(swapped_dict, file)


class DirectIndexing:
    def __init__(self, scenario: str, data: pd.DataFrame) -> None:
        self.data = data 
        self.scenario = scenario

        self.latitudes = data.loc[:, f"{scenario}_lat"]
        self.longitudes = data.loc[:, f"{scenario}_lng"]
        self.latitudes_index = data.columns.get_loc(f"{scenario}_lat")
        self.longitudes_index = data.columns.get_loc(f"{scenario}_lng")

        self.station_ids = data.loc[:, f"{scenario}_station_id"]
        self.station_names = data.loc[:, f"{scenario}_station_name"]
        self.station_id_index = data.columns.get_loc(f"{scenario}_station_id")
        self.station_name_index = data.columns.get_loc(f"{scenario}_station_name")

        self.number_of_rows = tqdm(iterable=range(data.shape[0]), desc="Searching through rows...")

    def found_rows_with_either_missing_ids_or_names(self) -> bool:
        """
        Search the rows of the dataset to find rows for where we have either a missing station name or a missing 
        station ID. In the version of the data that we are currently using, there no such rows.

        Returns:
            bool: a truth value indicating the existence or lack thereof of such rows. 
        """
        logger.info("Checking for rows that have either missing station names or station IDs")
        counter = 0
        for row in self.number_of_rows:
            station_id_for_row = self.data.iloc[row, self.station_id_index]
            station_name_for_row = self.data.iloc[row, self.station_name_index]
            
            if pd.isnull(station_name_for_row) and not pd.isnull(station_id_for_row) \
                or not pd.isnull(station_name_for_row) and pd.isnull(station_id_for_row):
                counter += 1

        return True if counter > 0 else False
    
    def find_rows_with_known_ids_and_names(self, save: bool = True) -> dict[str, tuple[float]]:
        """
        Find all the coordinates which have a known ID and known station name, and provide a dictionary of
        the respective rows and their associated coordinates.

        Returns:
            dict[str, tuple[float]]: pairs consisting of row numbers and their respective coordinates.
        """
        file_path = INDEXER_TWO/f"{self.scenario}_rows_and_coordinates_with_known_ids_names.json"
        
        if Path(file_path, mode="r").exists():
            logger.success("Fetching file containing each row with known station name ID, and its coordinates")
            with open(file_path, mode="r") as file:
                rows_and_coordinates_with_known_ids_names = json.load(file)

        else:
            logger.info("Looking for any row that has either a missing station ID OR a missing station name.")
            rows_and_coordinates_with_known_ids_names = {}

            for row in self.number_of_rows:

                latitude = self.data.iloc[row, self.latitudes_index]            
                longitude = self.data.iloc[row, self.longitudes_index]
    
                station_id_for_the_row = self.data.iloc[row, self.station_id_index]
                station_name_for_the_row = self.data.iloc[row, self.station_name_index]

                station_id_is_not_missing = not pd.isnull(station_id_for_the_row)
                station_name_is_not_missing = not pd.isnull(station_name_for_the_row)

                if station_id_is_not_missing and station_name_is_not_missing:
                    rows_and_coordinates_with_known_ids_names[row] = (latitude, longitude)

            if save:
                with open(file_path, mode="w") as file:
                    json.dump(rows_and_coordinates_with_known_ids_names, file)

        return rows_and_coordinates_with_known_ids_names 

    def match_names_and_ids_by_station_proximity(self, save: bool = True) -> dict[int, tuple[float]]:
        """
        Based on common sense and confirmation from https://account.divvybikes.com/map, it looks like there are 
        fingers crossed) no two stations that are within 10m of each other. On those grounds, we can declare that
        any two coordinates which are within 10m of each other must belong to the same station.

        Suppose we have a given coordinate (which we'll call the target coordinate), and we round it down from 6 to
        5 decimal places. If both coordinates of this rounded target coordinate are equal to the rounded version of 
        some other coordinate (on some other row) which has a known ID and known station name (we've confirmed that 
        it can't be one or the other), then the row of the target coordinate will assciated with the ID and station 
        name of the coordinate we found.
        """
        file_path = INDEXER_TWO/f"matched_{self.scenario}_coordinates_with_new_ids_and_names.json"

        if Path(file_path).exists():
            with open(file_path, mode="r") as file:
                rows_and_discovered_ids_and_names = json.load(file)

        else:
            assert not self.found_rows_with_either_missing_ids_or_names(), 'There is now a row which contains a \
                missing station ID or a station name (not both). This will have occured due to a change in the dataset'

            rows_and_discovered_ids_and_names = {}
            rows_and_coordinates_with_known_ids_names = self.find_rows_with_known_ids_and_names()
            logger.info("Rounding the coordinates in rows with both IDs and names")
            
            rows_and_known_rounded_coordinates = {}
            for row in tqdm(rows_and_coordinates_with_known_ids_names.keys()):
                    
                candidate_latitude: float = rows_and_coordinates_with_known_ids_names[row][0]
                candidate_longitude: float = rows_and_coordinates_with_known_ids_names[row][1]

                rounded_candidate_latitude = np.round(candidate_latitude, decimals=5)
                rounded_candidate_longitude = np.round(candidate_longitude, decimals=5)

                rows_and_known_rounded_coordinates[row] = (rounded_candidate_latitude, rounded_candidate_longitude)

            logger.info("Searching for rows with missing station IDs and names")

            rows_to_insert_names_and_ids = [
                row for row in tqdm(range(self.data.shape[0])) if pd.isnull(self.data.iloc[row, self.station_id_index]) and \
                    pd.isnull(self.data.iloc[row, self.station_name_index])
            ]

            counter = 0
            logger.info("Performing the matching operation...")
            for row in tqdm(rows_to_insert_names_and_ids):
                
                rounded_target_latitude = np.round(self.data.iloc[row, self.latitudes_index], decimals=5)
                rounded_target_longitude = np.round(self.data.iloc[row, self.longitudes_index], decimals=5)

                if (rounded_target_latitude, rounded_target_longitude) in rows_and_known_rounded_coordinates.values():

                    row_of_interest = next(
                        (int(row) for row, coordinate in rows_and_known_rounded_coordinates.items() if \
                            coordinate == (rounded_target_latitude, rounded_candidate_longitude)
                        )
                    )

                    print(type(row_of_interest))

                    found_station_id = self.data.iloc[row_of_interest, self.station_id_index]
                    found_station_name = self.data.iloc[row_of_interest, self.station_name_index]

                    rows_and_discovered_ids_and_names[row_of_interest] = (found_station_id, found_station_name)
                    counter += 1

                    print("Row", row_of_interest)
                    print("Rounded lat:", rounded_target_latitude)
                    print("Rounded lng:", rounded_target_latitude)

                    print("Found ID:", found_station_id)
                    print("Found Name:", found_station_name)

                    logger.success(f"Row #{row_of_interest} has one")

            if save:
                with open(INDEXER_TWO/f"matched_{self.scenario}_coordinates_with_new_ids_and_names.json", mode="w") as file:
                    json.dump(rows_and_discovered_ids_and_names, file)

            logger.success(f"Found {counter} station names and IDs")
            
        return rows_and_discovered_ids_and_names
    
    def replace_missing_station_names_and_ids(self) -> None:
        """
        We replace the missing station names and IDs in the dataframe with those that we discovered using our 
        matching procedure.
        """
        rows_with_new_names_and_ids: dict[int, tuple[float]] = self.match_names_and_ids_by_station_proximity()

        logger.info(f"Replacing {len(rows_with_new_names_and_ids)} missing IDs and names in the dataset")
        
        for row in rows_with_new_names_and_ids:
            new_station_id = rows_with_new_names_and_ids[row][0]
            new_station_name = rows_with_new_names_and_ids[row][1]

            self.data.replace(
                to_replace=self.data.loc[:, f"{self.scenario}_station_id"],
                value=new_station_id
            )

            self.data.replace(
                to_replace=self.data.loc[:, f"{self.scenario}_station_name"],
                value=new_station_name
            )

    def making_new_ids(self) -> dict[str, int]:

        unique_old_ids = self.data[f"{start_or_end}_station_id"].unique()
        new_ids = range(len(unique_old_ids))

        # Assign new station IDs
        old_ids_and_their_replacements = {
            unique_old_id: new_id for unique_old_id, new_id in zip(unique_old_ids, new_ids)
        }

        return old_ids_and_their_replacements


    def execute(self) -> None:

        old_ids_and_their_replacements = self.making_new_ids()

        for row in self.number_of_rows:
            for old_id, new_id in zip(old_ids_and_their_replacements.keys(), old_ids_and_their_replacements.values()):
                if old_id == self.data.loc[row, f"{self.station_id_index}"]:
                    self.data.replace(
                        to_replace=old_id, value=old_ids_and_their_replacements[old_id]
                    )



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



        
        

