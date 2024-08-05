import json
import random

from tqdm import tqdm
from loguru import logger
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd

from src.setup.paths import INDEXER_ONE, INDEXER_TWO, CLEANED_DATA
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
        self.scenario = scenario
        self.decimal_places = decimal_places
        self.data = data
 
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
        with open(INDEXER_ONE / f"rounded_{self.scenario}_points_and_new_ids.json", mode="w") as file:
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

        self.latitudes = self.data.loc[:, f"{scenario}_lat"]
        self.longitudes = self.data.loc[:, f"{scenario}_lng"]
        self.latitudes_index = self.data.columns.get_loc(f"{scenario}_lat")
        self.longitudes_index = self.data.columns.get_loc(f"{scenario}_lng")

        self.station_id_index = self.data.columns.get_loc(f"{scenario}_station_id")
        self.station_name_index = self.data.columns.get_loc(f"{scenario}_station_name")

        self.proper_name_of_scenario = "departure" if self.scenario == "start" else "arrival"

    def found_rows_with_either_missing_ids_or_names(self) -> bool:
        """
        Search the rows of the dataset to find rows for where we have either a missing station name or a missing 
        station ID. In the version of the data that we are currently using, there no such rows.

        Returns:
            bool: a truth value indicating the existence or lack thereof of such rows. 
        """
        counter = 0
        for row in tqdm(
                iterable=range(self.data.shape[0]),
                desc="Checking for rows that have either missing station names or station IDs"
        ):
            station_id_for_row = self.data.iloc[row, self.station_id_index]
            station_name_for_row = self.data.iloc[row, self.station_name_index]

            if pd.isnull(station_name_for_row) and not pd.isnull(station_id_for_row) \
                    or not pd.isnull(station_name_for_row) and pd.isnull(station_id_for_row):
                counter += 1

        return True if counter > 0 else False

    def find_rows_with_missing_ids_and_names(self, repeat: bool) -> list[int]:
        addendum = "still" if repeat else ""
        return [
            row for row in tqdm(
                iterable=range(self.data.shape[0]),
                desc=f"Searching for rows that {addendum} have missing station names and IDs"
            )

            if pd.isnull(self.data.iloc[row, self.station_id_index]) and
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
            rows_and_coordinates_with_known_ids_names = {}

            for row in tqdm(
                    iterable=range(self.data.shape[0]),
                    desc="Looking for rows that have either a missing station ID OR a missing station name."
            ):
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

    def match_names_and_ids_by_station_proximity(self, save: bool = True) -> dict[int, tuple[str, str]]:
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
        matched_coordinates_path = INDEXER_TWO / f"{self.scenario}_coordinates_with_new_ids_and_names.json"

        if Path(matched_coordinates_path).exists():
            logger.success("The matching operation has already been done. Fetching local file...")
            with open(matched_coordinates_path, mode="r") as file:
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

                rounded_latitude = np.round(original_coordinate[0], decimals=4)
                rounded_longitude = np.round(original_coordinate[1], decimals=4)
                rows_with_no_issue_and_their_rounded_coordinates[row] = (rounded_latitude, rounded_longitude)

            logger.info("Performing the matching operation...")

            for row in tqdm(
                    iterable=self.find_rows_with_missing_ids_and_names(repeat=False)
            ):

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
                with open(matched_coordinates_path, mode="w") as file:
                    json.dump(rows_with_the_issue_and_their_discovered_names_and_ids, file)

            logger.success(f"Found {len(rows_with_the_issue_and_their_discovered_names_and_ids)} station names and IDs")

        return rows_with_the_issue_and_their_discovered_names_and_ids

    def replace_missing_station_names_and_ids(self, save: bool = True) -> pd.DataFrame:
        """
        Take the row indices, as well as the associated IDs and names that were discovered using the
        matching procedure. Then replace the missing station names and IDs in these rows of the dataframe 
        with those that were discovered.
        """
        replaced_data_path = INDEXER_TWO / f"{self.scenario}_replaced_missing_names_and_ids.parquet"

        if Path(replaced_data_path).is_file():
            self.data = pd.read_parquet(replaced_data_path)

        else:
            rows_with_new_names_and_ids: dict[int, tuple[str, str]] = self.match_names_and_ids_by_station_proximity()

            # Write the target row indices, the new IDs, and the new names as vectors
            target_rows_indices = [int(row) for row in rows_with_new_names_and_ids.keys()]
            new_ids = {int(row): new_id_and_name[0] for row, new_id_and_name in rows_with_new_names_and_ids.items()}
            new_names = {int(row): new_id_and_name[1] for row, new_id_and_name in rows_with_new_names_and_ids.items()}
            
            # Perform the replacement
            self.data.iloc[target_rows_indices, self.station_id_index] = self.data.iloc[target_rows_indices, self.station_id_index].map(new_ids)
            self.data.iloc[target_rows_indices, self.station_name_index] = self.data.iloc[target_rows_indices, self.station_name_index].map(new_names)

            if save:
                self.data.to_parquet(path=replaced_data_path)

        return self.data

    def save_geodata(
        self, 
        station_names: pd.Series,
        station_ids: pd.Series,
        latitudes: pd.Series, 
        longitudes: pd.Series
        ) -> None:
        """
        Saves the station ID, mame, and coordinates for use in the frontend
        """
        geodata = {
            str(station_name): [(latitude, longitude), station_id] for (latitude, longitude, station_id, station_name) \
            in zip(latitudes, longitudes, station_ids, station_names)
        }

        with open(INDEXER_TWO / f"{self.scenario}_geodata.json", mode="w") as file:
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
        fully_cleaned_data_path = CLEANED_DATA / f"fully_cleaned_{self.scenario}s.parquet"

        if Path(fully_cleaned_data_path).is_file():
            logger.success("Data with completely re-indexed station IDs already exists. Fetching it...")
            self.data = pd.read_parquet(path=fully_cleaned_data_path)

        else:
            logger.info("Initiating reindexing procedure...")
            self.data = self.replace_missing_station_names_and_ids()
            leftover_rows = self.find_rows_with_missing_ids_and_names(repeat=True)

            if delete_leftover_rows:
                logger.info("Deleting the leftover rows...")
                self.data = self.data.drop(self.data.index[leftover_rows], axis=0)

            else:
                logger.info("Initiating the reverse geocoding procedure for the leftover rows")
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
            old_and_new_ids = {old_id: index for index, old_id in enumerate(unique_old_ids)}
            self.data.iloc[:, self.station_id_index] = station_ids.map(old_and_new_ids)

            self.data = self.data.reset_index(drop=True)

            self.save_geodata(
                latitudes=self.data.iloc[:, self.latitudes_index],
                longitudes=self.data.iloc[:, self.longitudes_index],
                station_ids=self.data.iloc[:, self.station_id_index],
                station_names=self.data.iloc[:, self.station_name_index],
            )

            self.data = self.data.drop(
                columns=[f"{self.scenario}_lat", f"{self.scenario}_lat", f"{self.scenario}_station_name"]
            )

            if save:
                self.data.to_parquet(path=fully_cleaned_data_path)

        return self.data
