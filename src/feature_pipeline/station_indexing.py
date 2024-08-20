import json
import random

from tqdm import tqdm
from loguru import logger
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd

from src.setup.paths import INDEXER_TWO, INDEXER_TWO, CLEANED_DATA
from src.feature_pipeline.feature_engineering import ReverseGeocoding


class RoundingCoordinates:
    """
    The contents of this class are only to be used if the data being processed is so voluminous that it 
    poses a such a problem (in terms of memory and time) that the compromise of geographical accuracy 
    resulting from its use can be justified.
    """

    def __init__(self, data: pd.DataFrame, scenario: str, decimal_places: int | None) -> None:
        """
        Args:
            decimal_places (int): the number of decimal places to which we will round the coordinates. 
                                The original coordinates are written in 6 decimal places. For each 
                                decimal place that is lost, the accuracy of the coordinates degrades 
                                by a factor of 10 meters

            scenario (str): whether we are looking at departures ("start") or arrivals ("end").
        """
        self.data = data
        self.scenario = scenario
        self.decimal_places = decimal_places
 
    def add_column_of_rounded_coordinates_to_dataframe(self) -> pd.DataFrame:
        """
        This function takes the latitude and longitude columns of a dataframe, rounds them down to a 
        specified number of decimal places, and makes a column which consists of points containing the 
        rounded latitudes and longitudes.
        """
        logger.info(f"Approximating the coordinates of the location where each trip {self.scenario}s...")

        new_latitudes = []
        new_longitudes = []

        for latitude in tqdm(iterable=self.data[f"{self.scenario}_lat"].values, desc="Working on latitudes"):
            new_latitudes.append(
                np.round(latitude, decimals=self.decimal_places)
            )

        for longitude in tqdm(iterable=self.data[f"{self.scenario}_lng"].values, desc="Working on longitudes"):
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
        self.data = self.data.drop(
            columns=[f"{self.scenario}_lat", f"{self.scenario}_lng"]
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
        self.data = self.data.drop(
            columns=[f"rounded_{self.scenario}_lat", f"rounded_{self.scenario}_lng"]
        )

        return self.data

    def make_station_ids_from_unique_coordinates(self) -> dict[float, int]:
        """
        This function makes a list of random numbers for each unique point, and 
        associates each point with a corresponding number. This effectively creates new 
        IDs for each location.
        """
        logger.info("Matching up approximate locations with generated IDs...")

        unique_coordinates = self.data[f"rounded_{self.scenario}_points"].unique()
        num_unique_points = len(unique_coordinates)

        # Set a seed to ensure reproducibility. 
        random.seed(69)

        # Make a random mixture of the numbers from 0 to len(num_unique_points) 
        new_station_ids = random.sample(population=range(num_unique_points), k=num_unique_points)

        # Make a dictionary of points
        points_and_new_ids = {}

        for point, new_station_id in tqdm(zip(unique_coordinates, new_station_ids)):
            points_and_new_ids[point] = new_station_id

        # Because tuples can't be keys of a dictionary
        swapped_dict = {station_id: point for point, station_id in points_and_new_ids.items()}
        with open(INDEXER_TWO / f"rounded_{self.scenario}_points_and_new_ids.json", mode="w") as file:
            json.dump(swapped_dict, file)

        return points_and_new_ids

    def execute(self) -> pd.DataFrame:
        """
        Take each point, and the ID which corresponds to it (within its dictionary),
        and put those IDs in the relevant dataframe (in a manner that matches each 
        point with its ID row-wise).
        """
        self.data = self.data.drop(f"{self.scenario}_station_id", axis=1)
        self.data = self.add_column_of_rounded_coordinates_to_dataframe()

        points_and_ids = self.make_station_ids_from_unique_coordinates()

        new_station_ids = [
            points_and_ids[point] for point in list(self.data.loc[:, f"rounded_{self.scenario}_points"]) if
            point in points_and_ids.keys()
        ]

        self.data.insert(
            loc=self.data.shape[1],
            column=f"{self.scenario}_station_id",
            value=pd.Series(new_station_ids),
            allow_duplicates=False
        )

        return self.data


class DirectIndexing:
    def __init__(self, scenario: str, data: pd.DataFrame) -> None:
        self.data = data
        self.scenario = scenario

        self.latitudes_index = data.columns.get_loc(f"{scenario}_lat")
        self.longitudes_index = data.columns.get_loc(f"{scenario}_lng")
        self.station_id_index = data.columns.get_loc(f"{scenario}_station_id")
        self.station_name_index = data.columns.get_loc(f"{scenario}_station_name")

        self.station_ids = data.iloc[:, self.station_id_index]
        self.station_names = data.iloc[:, self.station_name_index]
        self.proper_name_of_scenario = "departure" if scenario == "start" else "arrival"

    def found_rows_with_either_missing_ids_or_names(self) -> bool:
        """
        Search the dataset for rows which have either a missing station name or a missing 
        station ID. In the version of the data that we are currently using, there no such rows.

        Returns:
            bool: a truth value indicating the existence or lack thereof of such rows. 
        """
        logger.info("Checking for rows that either have missing station names or missing IDs")

        # Boolean pandas series 
        only_missing_id = self.station_names.notnull() & self.station_ids.isnull()
        only_missing_names = self.station_names.isnull() & self.station_ids.notnull()

        target_condition = only_missing_id | only_missing_names
        return True if target_condition.sum() > 0 else False

    @staticmethod
    def find_rows_with_missing_ids_and_names(data: pd.DataFrame, scenario: str, first_time: bool) -> list[int]:
        """
        Search for rows with both IDs and names missing.

        Args:
            first_time (bool): whether this function is being run for the first time

        Returns:
            list[int]: the indices of the rows we found.
        """
        logger.info(f"Searching for rows that{"" if first_time else " still"} have missing station names and IDs...")

        missing_station_ids = data[f"{scenario}_station_id"].isnull()
        missing_station_names = data[f"{scenario}_station_name"].isnull()

        mask_of_problem_rows = missing_station_ids & missing_station_names
        problem_rows = data.loc[mask_of_problem_rows, :]
        return problem_rows.index

    def find_rows_with_known_ids_and_names(self) -> dict[str, tuple[float]]:
        """
        Find all the coordinates which have a known ID and known station name, and provide a dictionary 
        of the respective rows and their associated coordinates.

        Returns:
            dict[str, tuple[float]]: pairs consisting of row numbers and their respective coordinates.
        """
        logger.info("Looking for rows that have both station names and IDs...")

        present_station_ids = self.station_ids.notnull()
        present_station_names = self.station_names.notnull()
        complete_rows_mask = present_station_ids & present_station_names
        complete_rows = self.data.loc[complete_rows_mask, :]
        
        latitudes_of_complete_rows = self.data.loc[complete_rows_mask, f"{self.scenario}_lat"]
        longitudes_of_complete_rows = self.data.loc[complete_rows_mask,  f"{self.scenario}_lng"]

        rows_and_coordinates_with_known_ids_names = dict(
            zip(
                complete_rows.index, zip(latitudes_of_complete_rows, longitudes_of_complete_rows)
            )
        )

        return rows_and_coordinates_with_known_ids_names

    def match_names_and_ids_by_station_proximity(self) -> dict[int, tuple[str, str]]:
        """
        Based on common sense, and a perfunctory look at https://account.divvybikes.com/map, it looks like there 
        are (knock on wood) no two stations that are within 10m of each other. On those grounds, we can say with some 
        confidence that any two station coordinates which are within 10m of each other must belong to the same station.

        Suppose we have a given coordinate (which we'll call the target coordinate), and we round it down from 6 to
        4 decimal places. If both coordinates of this rounded target coordinate are equal to the rounded version of 
        some other coordinate (on some other row) which has a known ID and known station name (we've confirmed that 
        it can't be one or the other), then the row of the target coordinate will be associated with the ID and station
        name of the coordinate we found.

        Returns:
            dict[int, tuple[str|int, str]]: key, value pairs of row indices and their newly discovered station IDs
                                            and names
        """
        assert not self.found_rows_with_either_missing_ids_or_names(), 'There is now a row which contains a \
            missing station ID or a station name (not both). This will have occurred due to a change in the data'

        logger.info("Starting the matching process...")

        complete_rows_and_their_original_coordinates = self.find_rows_with_known_ids_and_names()
        coordinates_of_complete_rows = np.array(list(complete_rows_and_their_original_coordinates.values()))
        rounded_coordinates_of_complete_rows = np.round(coordinates_of_complete_rows, decimals=4)

        problem_rows_indices = self.find_rows_with_missing_ids_and_names(
            data=self.data,
            scenario=self.scenario, 
            first_time=True
        )

        complete_row_indices = np.array([int(index) for index in complete_rows_and_their_original_coordinates.keys()])

        complete_rows_and_their_rounded_coordinates = {
            row_index: tuple(rounded_coordinates_of_complete_rows[i]) for i, row_index in enumerate(complete_row_indices)
        }

        rounded_problem_lats = np.round(self.data.iloc[problem_rows_indices, self.latitudes_index].values, decimals=4)
        rounded_problem_lngs = np.round(self.data.iloc[problem_rows_indices, self.longitudes_index].values, decimals=4)
        rounded_problem_coordinates = list(zip(rounded_problem_lats, rounded_problem_lngs))

        # Get a boolean array of the indices of rounded coordin
        rounded_coordinates_match = np.isin(rounded_problem_coordinates, rounded_coordinates_of_complete_rows).all(axis=1)

        complete_rows_with_matches = complete_row_indices[np.where(rounded_coordinates_match)[0]]  
                    
        is_problem_row = np.isin(element=problem_rows_indices, test_elements=np.arange(len(self.data)))      
        rows_to_be_targeted = np.where(is_problem_row & rounded_coordinates_match)[0]

        found_ids  = self.data.iloc[rows_to_be_targeted, self.station_id_index]
        found_names  = self.data.iloc[rows_to_be_targeted, self.station_name_index]

        problem_rows_and_their_discovered_names_and_ids = {
            int(index): (code, name) for index, code, name in zip(rows_to_be_targeted, found_ids, found_names)
        }

        logger.success(f"Found {len(problem_rows_and_their_discovered_names_and_ids)} station names and IDs")
        return problem_rows_and_their_discovered_names_and_ids

    def replace_missing_station_names_and_ids(self) -> pd.DataFrame:
        """
        Take the row indices, as well as the associated IDs and names that were discovered using the matching 
        procedure. Then replace the missing station names and IDs in these rows of the dataframe with those 
        that were discovered.

        Args:
            save (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        rows_with_new_names_and_ids: dict[int, tuple[str, str]] = self.match_names_and_ids_by_station_proximity()

        # Write the target row indices, the new IDs, and the new names as vectors
        target_rows_indices = [int(row) for row in rows_with_new_names_and_ids.keys()]

        new_ids = {
            int(row): new_id for row, (new_id, new_name) in rows_with_new_names_and_ids.items()
        }

        new_names = {
            int(row): new_name for row, (new_id, new_name) in rows_with_new_names_and_ids.items()
        }
        
        # Perform the replacement
        self.data.iloc[target_rows_indices, self.station_id_index] = \
            self.data.iloc[target_rows_indices, self.station_id_index].map(new_ids)
        
        self.data.iloc[target_rows_indices, self.station_name_index] = \
            self.data.iloc[target_rows_indices, self.station_name_index].map(new_names)

        return self.data

    @staticmethod
    def save_geodata(data: pd.DataFrame, scenario: str, for_plotting: bool) -> None:
        """
        Saves the station ID, mame, and coordinates for use in the frontend
        """
        station_names = data[f"{scenario}_station_name"].values
        station_ids = data[f"{scenario}_station_id"].values
        longitudes = data[f"{scenario}_lng"].values
        latitudes = data[f"{scenario}_lat"].values

        geodata_to_iterate = tqdm(
            iterable=zip(latitudes, longitudes, station_ids, station_names),
            desc="Saving the geodata in each row"
        )

        if for_plotting:
            file_path = INDEXER_TWO / f"{scenario}_geojson.geojson"

            geodata = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",

                        "geometry": {
                            "type": "Point",
                            "coordinate": [longitude, latitude]  # Apparently, this reversal is standard for geojson
                        },

                        "properties": {
                            "station_id": station_id,
                            "station_name": station_name
                        }
                    } 
                    for (latitude, longitude, station_id, station_name) in geodata_to_iterate      
                ] 
            }

        else:
            file_path = INDEXER_TWO / f"{scenario}_geodata.json"
            geodata = [
                {   
                    "coordinates": [latitude, longitude],
                    "station_id": station_id,
                    "station_name": station_name    
                } for latitude, longitude, station_id, station_name in 
                    zip(latitudes, longitudes, station_ids, station_names)
            ]

        with open(file_path, mode="w") as file:
            json.dump(geodata, file)

    def execute(self, delete_leftover_rows: bool = True, save: bool = True) -> pd.DataFrame:
        """
        Make a replacement for every existing ID because many of the IDs are long strings (see the preprocessing
        script for details).

        Args:
            delete_leftover_rows:
            save:

        Returns:
            pd.DataFrame: the data, but with all the station IDs re-indexed
        """
        logger.info("Initiating reindexing procedure for the station IDs...")

        leftover_rows = self.find_rows_with_missing_ids_and_names(
            data=self.data, 
            scenario=self.scenario,
            first_time=False
        )

        if delete_leftover_rows:
            logger.warning(f"Deleting the {len(leftover_rows)} rows that still have no station IDs and names.")
            self.data = self.data.drop(self.data.index[leftover_rows], axis=0)

        else:
            logger.info("Initiating reverse geocoding procedure for the leftover rows")
            coordinate_maker = RoundingCoordinates(decimal_places=6, scenario=self.scenario, data=self.data)
            coordinate_maker.add_column_of_rounded_coordinates_to_dataframe(scenario=self.scenario, data=self.data)

            for column in self.data.columns:
                if column not in [f"{self.scenario}_station_id", f"rounded_{self.scenario}_points"]:
                    self.data = self.data.drop(column, axis=1)

            self.data.rename(
                columns={f"rounded_{self.scenario}_points": "coordinates"}
            )

            reverse_geocoder = ReverseGeocoding(scenario=self.scenario, geodata=self.data)

            # TO DO: COMPLETE THIS PROCEDURE

        station_ids = self.data.iloc[:, self.station_id_index]
        unique_old_ids = station_ids.unique()
        
        # Use the indices of this enumerate as the new station IDs
        old_and_new_ids = {old_id: index for index, old_id in enumerate(unique_old_ids)}
        self.data.iloc[:, self.station_id_index] = station_ids.map(old_and_new_ids)
        self.data = self.data.reset_index(drop=True)

        for column in self.data.select_dtypes(include=["datetime64[ns]"]):
            self.data[column] = self.data[column].astype(str)

        self.save_geodata(data=self.data, scenario=self.scenario, for_plotting=False)
        self.save_geodata(data=self.data, scenario=self.scenario, for_plotting=True)

        self.data = self.data.drop(
            columns=[f"{self.scenario}_lat", f"{self.scenario}_lng", f"{self.scenario}_station_name"]
        )

        if save:
            self.data.to_parquet(path=CLEANED_DATA / f"fully_cleaned_and_reindexed_{self.scenario}_data.parquet")

        return self.data
