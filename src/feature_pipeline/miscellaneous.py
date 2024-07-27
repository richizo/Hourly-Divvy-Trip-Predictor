# Utilities
import json
import random

from tqdm import tqdm
from loguru import logger
from pathlib import Path, PosixPath

# Data Manipulation and Access
import numpy as np
import pandas as pd

# Custom modules
from src.setup.paths import INDEXER_TWO
from src.feature_pipeline.feature_engineering import ReverseGeocoding

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
        self.decimal_places = decimal_places

    def add_column_of_rounded_coordinates_to_dataframe(self, scenario: str, data: pd.DataFrame) -> None:
        """
        This function takes the latitude and longitude columns of a dataframe, rounds them down to a 
        specified number of decimal places, and makes a column which consists of points containing the 
        rounded latitudes and longitudes.
        """
        new_latitudes = []
        new_longitudes = []

        latitudes = tqdm(
            iterable=data[f"{scenario}_lat"].values,
            desc="Working on latitudes"
        )

        longitudes = tqdm(
            iterable=data[f"{scenario}_lng"].values,
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
            data.insert(
                loc=data.shape[1],
                column=f"rounded_{scenario}_{coordinate_name}",
                value=pd.Series(coordinate),
                allow_duplicates=False
            )

        # Remove the original latitudes and longitudes
        data = data.drop(
            columns=[f"{scenario}_lat", f"{scenario}_lng"]
        )

        rounded_points = list(
            zip(
                data[f"rounded_{scenario}_lat"], data[f"rounded_{scenario}_lng"]
            )
        )

        # Insert the rounded() coordinates as points
        data.insert(
            loc=data.shape[1],
            column=f"rounded_{scenario}_points",
            value=pd.Series(rounded_points),
            allow_duplicates=False
        )

        # Remove the rounded latitudes and longitudes that we added
        data = data.drop(
            columns=[f"rounded_{scenario}_lat", f"rounded_{scenario}_lng"],
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

    def add_column_of_ids(self, data: pd.DataFrame, scenario: str, points_and_ids: dict) -> None:
        """
        Take each point, and the ID which corresponds to it (within its dictionary),
        and put those IDs in the relevant dataframe (in a manner that matches each 
        point with its ID row-wise).

        Args:
            points_and_ids (dict): dictionary of unique coordinates and IDs.
        """
        station_ids = [
            points_and_ids[point] for point in list(data.loc[:, f"rounded_{scenario}_points"]) if
            point in points_and_ids.keys()
        ]

        data.insert(
            loc=data.shape[1],
            column=f"{scenario}_station_id",
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

        self.number_of_rows = tqdm(iterable=range(data.shape[0]))
        self.matched_coordinates_path = INDEXER_TWO/f"matched_{self.scenario}_coordinates_with_new_ids_and_names.json"

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

    def find_rows_with_missing_ids_and_names(self) -> list[int]:
            
        logger.info("Searching for rows with missing station names and IDs")
        
        return [
            row for row in self.number_of_rows if pd.isnull(self.data.iloc[row, self.station_id_index]) and
            pd.isnull(self.data.iloc[row, self.station_name_index])
        ]

    def find_rows_with_known_ids_and_names(self, save: bool = True) -> dict[str, tuple[float]]:
        """
        Find all the coordinates which have a known ID and known station name, and provide a dictionary of
        the respective rows and their associated coordinates.

        Returns:
            dict[str, tuple[float]]: pairs consisting of row numbers and their respective coordinates.
        """
        file_path = INDEXER_TWO / f"{self.scenario}_rows_and_coordinates_with_known_ids_names.json"

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

    def match_names_and_ids_by_station_proximity(self, save: bool = True) -> dict[int, tuple[str | int, str]]:
        """
        Based on common sense, and a perfunctory look at https://account.divvybikes.com/map, it looks like there are 
        (fingers crossed) no two stations that are within 10m of each other. On those grounds, we can say with some 
        confidence that any two station coordinates which are within 10m of each other must belong to the same station.

        Suppose we have a given coordinate (which we'll call the target coordinate), and we round it down from 6 to
        4 decimal places. If both coordinates of this rounded target coordinate are equal to the rounded version of 
        some other coordinate (on some other row) which has a known ID and known station name (we've confirmed that 
        it can't be one or the other), then the row of the target coordinate will be associated with the ID and station
        name of the coordinate we found.

        Args:
            save (bool, optional): whether to save the final output. Defaults to True.

        Returns:
            dict[int, tuple[str|int, str]]: key, value pairs of row indices and their newly discovered station IDs
                                            and names
        """
        if Path(self.matched_coordinates_path).exists():
            logger.success("The matching operation has already been done. Fetching local file...")
            with open(self.matched_coordinates_path, mode="r") as file:
                rows_with_the_issue_and_their_discovered_names_and_ids = json.load(file)

        else:
            assert not self.found_rows_with_either_missing_ids_or_names(), 'There is now a row which contains a \
                missing station ID or a station name (not both). This will have occurred due to a change in the data'

            rows_with_the_issue_and_their_discovered_names_and_ids = {}
            rows_with_no_issue_and_their_original_coordinates = self.find_rows_with_known_ids_and_names()
            logger.info("Rounding the coordinates in rows with both IDs and names")

            rows_with_no_issue_and_their_rounded_coordinates = {}
            for row in tqdm(rows_with_no_issue_and_their_original_coordinates.keys()):
                original_coordinate = rows_with_no_issue_and_their_original_coordinates[row]

                rounded_candidate_latitude = np.round(original_coordinate[0], decimals=4)
                rounded_candidate_longitude = np.round(original_coordinate[1], decimals=4)

                rows_with_no_issue_and_their_rounded_coordinates[row] = (
                    rounded_candidate_latitude, rounded_candidate_longitude)

            logger.info("Performing the matching operation...")
            for row in tqdm(self.rows_with_the_issue):

                if row in rows_with_the_issue_and_their_discovered_names_and_ids.keys():
                    continue

                rounded_target_latitude = np.round(self.data.iloc[row, self.latitudes_index], decimals=4)
                rounded_target_longitude = np.round(self.data.iloc[row, self.longitudes_index], decimals=4)

                if (rounded_target_latitude, rounded_target_longitude) in \
                        rows_with_no_issue_and_their_rounded_coordinates.values():
                    row_of_interest = next(
                        (
                            int(row) for row, coordinate in rows_with_no_issue_and_their_rounded_coordinates.items() if
                            coordinate == (rounded_target_latitude, rounded_target_longitude)
                        )
                    )

                    found_station_id = self.data.iloc[row_of_interest, self.station_id_index]
                    found_station_name = self.data.iloc[row_of_interest, self.station_name_index]

                    rows_with_the_issue_and_their_discovered_names_and_ids[row] = (found_station_id, found_station_name)

            if save:
                with open(self.matched_coordinates_path, mode="w") as file:
                    json.dump(rows_with_the_issue_and_their_discovered_names_and_ids, file)

            logger.success(f"Found {len(rows_with_the_issue_and_their_discovered_names_and_ids)} station names and IDs")

        return rows_with_the_issue_and_their_discovered_names_and_ids

    def replace_missing_station_names_and_ids(self, save: bool = True) -> pd.DataFrame:
        """
        Take the row indices, as well as the associated IDs and names that were discovered using the
        matching procedure. Then replace the missing station names and IDs in these rows of the dataframe 
        with those that were discovered.
        """
        replaced_data_path = INDEXER_TWO/f"{self.scenario}_replaced_missing_names_and_ids.parquet"

        if Path(replaced_data_path).exists():    
            self.data = pd.read_parquet(replaced_data_path)

        else:
            rows_with_new_names_and_ids: dict[int, tuple[str | int, str]] = self.match_names_and_ids_by_station_proximity()

            rows_to_replace = tqdm(
                iterable=rows_with_new_names_and_ids.keys(),
                desc=f"Replacing missing IDs and names in the dataset"
            )

            for row in rows_to_replace:

                new_station_id: str = rows_with_new_names_and_ids[row][0]
                new_station_name: str = rows_with_new_names_and_ids[row][1]

                # Because the row indices in the dictionary are strings, and I don't want the .iloc method to complain.
                row = int(row)  

                if row <= len(self.data):

                    # These are of type "None", however the replace method of the dataframe class complains if I try to
                    # replace an object of type "None". So we'll just wrap the objects in their proper types to shut it up.
                    empty_station_id = str(self.data.iloc[row, self.station_id_index]) 
                    empty_station_name = str(self.data.iloc[row, self.station_name_index])

                    self.data.replace(to_replace=empty_station_id, value=new_station_id)
                    self.data.replace(to_replace=empty_station_id, value=new_station_name)

                else:
                    logger.error(f"Row {row} is not part of the dataset. It was probably removed during an earlier process")

            if save:
                self.data.to_parquet(path=replaced_data_path)

        return self.data

    def make_and_insert_new_ids(
        self, 
        delete_leftover_rows: bool | None,
        reverse_geocode: bool | None,
    ) -> pd.DataFrame:
        """
        Make a new ID for every existing ID, and replace the existing IDs with their replacements.

        Returns:
            pd.DataFrame: 
        """
        assert delete_leftover_rows or reverse_geocode, "You must either choose to delete the leftover rows, or \
            use reverse geocoding to name their stations/generate IDs"

        self.data = self.replace_missing_station_names_and_ids()
        leftover_rows = self.find_rows_with_missing_ids_and_names()

        if delete_leftover_rows:
            self.data = self.data.drop(self.data.index[leftover_rows], axis=0) 

        elif reverse_geocode:
            coordinate_maker = RoundingCoordinates(decimal_places=6)
            coordinate_maker.add_column_of_rounded_coordinates_to_dataframe(scenario=self.scenario, data=self.data)

            for column in self.data.columns:
                if column not in [f"{self.scenario}_station_id", f"rounded_{scenario}_points"]:
                    self.data = self.data.drop(column, axis = 1)

            self.data.rename(
                columns={f"rounded_{self.scenario}_points": "coordinates"}
            )

            reverse_geocoder = ReverseGeocoding(scenario=self.scenario, geodata=self.data)

            # TO DO: COMPLETE THIS PROCEDURE

        unique_old_ids = self.data[f"{self.scenario}_station_id"].unique()
        new_ids = range(len(unique_old_ids))

        # Assign new station IDs
        old_ids_and_their_replacements: dict[str | int, int] = {
            unique_old_id: new_id for unique_old_id, new_id in zip(unique_old_ids, new_ids)
        }


        logger.info("Replacing original station IDs with new ones...")
        for row in self.number_of_rows:

            for old_id, new_id in zip(unique_old_ids, new_ids):
                if old_id == self.data.loc[row, f"{self.station_id_index}"]:
                    self.data.replace(
                        to_replace=old_id, value=old_ids_and_their_replacements[old_id]
                    )

        return self.data


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
