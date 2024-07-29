# Utilities 
import os
import warnings
from tqdm import tqdm
from loguru import logger
from pathlib import Path

# Data Manipulation & Access
import numpy as np
import pandas as pd

# Custom code 
from src.setup.config import config
from src.feature_pipeline.data_extraction import load_raw_data
from src.feature_pipeline.station_indexing import RoundingCoordinates, DirectIndexing
from src.feature_pipeline.feature_engineering import perform_feature_engineering

from src.setup.paths import (
    CLEANED_DATA, TRAINING_DATA, TIME_SERIES_DATA, INDEXER_ONE, INDEXER_TWO, INFERENCE_DATA, make_fundamental_paths
)


class DataProcessor:
    def __init__(self, year: int, for_inference: bool):
        self.station_ids = None
        self.indexer = None
        self.scenarios = ["start", "end"]
        self.start_ts_path = TIME_SERIES_DATA / "start_ts.parquet"
        self.end_ts_path = TIME_SERIES_DATA / "end_ts.parquet"

        self.data = pd.concat(list(load_raw_data(year=year))) if not for_inference else None

    @staticmethod
    def use_custom_station_indexing(scenarios: list[str], data: pd.DataFrame) -> bool:
        """
        Certain characteristics of the data will lead to the selection of one of two custom methods of indexing 
        the station IDs. This is necessary because there are several station IDs such as "KA1504000135" (dubbed 
        long IDs) which we would like to be rid of.  We observe that the vast majority of the IDs contain no more 
        than 6 or 7 values, while the long IDs are generally longer than 7 characters. The second group of 
        problematic station IDs are, naturally, the missing ones.
        
        This function starts by checking which how many of the station IDs fall into either of these two groups. 
        If there are enough of these (subjectively determined to be at least half the total number of IDs), then 
        the station IDs will have to be replaced with numerical values using one of the two custom indexing methods
        mentioned before. Which of these methods will be used will depend on the result of the function after this 
        one.

        Returns:
            bool: whether a custom indexing method will be used. 
        """
        results = []

        for scenario in scenarios:

            long_id_counter = 0
            for station_id in data.loc[:, f"{scenario}_station_id"]:
                if len(str(station_id)) > 7 and not pd.isnull(station_id):
                    long_id_counter += 1

            num_missing_indices = data[f"{scenario}_station_id"].isna().sum()
            result = True if (num_missing_indices + long_id_counter) / data.shape[0] >= 0.5 else False
            results.append(result)

        return True if False not in results else False

    @staticmethod
    def tie_ids_to_unique_coordinates(data: pd.DataFrame) -> bool:
        """
        With a large enough dataset (subjectively determined to be those that has more than 10M rows), the author has 
        finds it necessary to round the coordinates of each station to make the preprocessing operations that follow 
        less taxing in terms of time and system memory. Following the rounding operation, a number will be assigned to 
        each unique coordinate which will function as the new ID for that station. 

        In smaller datasets (which are heavily preferred by the author), the new IDs are created with no connection to
        the number of unique coordinates.

        Args:
            data (pd.DataFrame): the dataset to be examined.

        Returns:
            bool: whether the dataset is deemed to be large enough to trigger the 
        """
        return True if len(data) > 10_000_000 else False

    def make_training_data(self, geocode: bool) -> list[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data, clean it, transform it into time series data, and transform that in turn into
        training data which is subsequently saved.

        Args:
            geocode (bool): whether to geocode as part of feature engineering.
            
        Returns:
            list[pd.DataFrame]: a list containing the datasets for the starts and ends of trips.
        """
        start_ts, end_ts = self.make_time_series()
        ts_data_per_scenario = {"start": start_ts, "end": end_ts}

        training_sets = []
        for scenario in ts_data_per_scenario.keys():
            logger.info(f"Turning the time series data on the {scenario}s of trips into training data...")

            training_data = self.transform_ts_into_training_data(
                ts_data=ts_data_per_scenario[scenario],
                for_inference=False,
                geocode=geocode,
                scenario=scenario,
                input_seq_len=config.n_features,
                step_size=24
            )

            training_sets.append(training_data)

        return training_sets

    def make_time_series(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform the transformation of the raw data into time series data 

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets for departures and arrivals respectively.
        """
        logger.info("Cleaning dataframe")
        self.data = self.clean()

        start_df_columns = ["started_at", "start_lat", "start_lng", "start_station_id"]
        end_df_columns = ["ended_at", "end_lat", "end_lng", "end_station_id"]

        if self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) and not \
                self.tie_ids_to_unique_coordinates(data=self.data):

            start_df_columns.append("start_station_name")
            end_df_columns.append("end_station_name")

        start_df = self.data[start_df_columns]
        end_df = self.data[end_df_columns]

        logger.info("Transforming the data into a time series...")
        start_ts, end_ts = self.transform_cleaned_data_into_ts_data(start_df=start_df, end_df=end_df)
        return start_ts, end_ts

    def clean(self, save: bool = True) -> pd.DataFrame:

        if self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) \
                and self.tie_ids_to_unique_coordinates(data=self.data):

            cleaned_data_file_path = CLEANED_DATA / "cleaned_data_indexer_one.parquet"

        elif self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) \
                and not self.tie_ids_to_unique_coordinates(data=self.data):

            cleaned_data_file_path = CLEANED_DATA / "partially_cleaned_data_indexer_two.parquet"

        else:
            raise NotImplementedError(
                "The majority of Divvy's IDs weren't numerical and valid during initial development."
            )

        if not Path(cleaned_data_file_path).is_file():

            self.data["started_at"] = pd.to_datetime(self.data["started_at"], format="mixed")
            self.data["ended_at"] = pd.to_datetime(self.data["ended_at"], format="mixed")

            def _delete_rows_with_missing_station_names_and_coordinates() -> pd.DataFrame:
                """
                There are rows with missing latitude and longitude values for the various
                stations. If any of these rows have available station names, then geocoding
                can be used to get the coordinates. At the current time however, all rows with
                missing coordinates also have missing station names, rendering these rows
                irreparably lacking. 
                
                We locate and delete these points with this function.

                Returns:
                    pd.DataFrame: the data, absent the aforementioned rows.
                """
                for scenario in self.scenarios:
                    lats = self.data.columns.get_loc(f"{scenario}_lat")
                    longs = self.data.columns.get_loc(f"{scenario}_lng")
                    station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                    all_rows = tqdm(
                        iterable=range(self.data.shape[0]),
                        desc=f"Targeting rows with missing station names and coordinates for deletion ({scenario}s)"
                    )

                    rows_to_delete = []
                    for row in all_rows:
                        station_name = self.data.iloc[row, station_names_col]
                        row_latitude = self.data.iloc[row, lats]
                        row_longitude = self.data.iloc[row, longs]

                        if pd.isnull(station_name) and pd.isnull(row_latitude) and pd.isnull(row_longitude):
                            rows_to_delete.append(row)

                    self.data = self.data.drop(self.data.index[rows_to_delete], axis=0)

                return self.data

            self.data = _delete_rows_with_missing_station_names_and_coordinates()
            features_to_drop = ["ride_id", "rideable_type", "member_casual"]

            if self.use_custom_station_indexing(data=self.data, scenarios=self.scenarios) and \
                    self.tie_ids_to_unique_coordinates(data=self.data):
                features_to_drop.extend(
                    ["start_station_id", "start_station_name", "end_station_id", "end_station_name"]
                )

            self.data = self.data.drop(columns=features_to_drop)

            if save:
                self.data.to_parquet(path=cleaned_data_file_path)

            return self.data

        else:
            logger.success("There is already some cleaned data. Fetching it...")
            return pd.read_parquet(path=cleaned_data_file_path)

    def transform_cleaned_data_into_ts_data(
            self,
            start_df: pd.DataFrame,
            end_df: pd.DataFrame,
            save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts cleaned data into time series data.

        In addition to the putting the arrival and departure times in hourly form, we approximate
        the latitudes and longitudes of each point of origin or destination (we are targeting
        no more than a 100m radius of each point, but ideally we would like to maintain a 10m
        radius), and use these to construct new station IDs.

        Args:
            start_df (pd.DataFrame): dataframe of departure data

            end_df (pd.DataFrame): dataframe of arrival data

            save (bool): whether we wish to save the generated time series data

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on arrivals or departures
        """

        def _get_ts_or_begin_transformation(start_ts_path: str, end_ts_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

            if Path(start_ts_path).exists() and Path(end_ts_path).exists():
                logger.success("Both time series datasets are already present")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif not Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("Neither time series dataset exists")
                start_ts, end_ts = _begin_transformation(missing_scenario="both")
                return start_ts, end_ts

            elif not Path(start_ts_path).exists() and Path(end_ts_path).exists():
                logger.warning("Time series data for departures has not been made")
                start_ts = _begin_transformation(missing_scenario="start")
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("Time series data for arrivals has not been made")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = _begin_transformation(missing_scenario="end")
                return start_ts, end_ts

        def _begin_transformation(missing_scenario: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:

            dictionaries: list[dict] = []
            interim_dataframes: list[pd.DataFrame] = []

            def __investigate_making_new_station_ids(
                    cleaned_data: pd.DataFrame,
                    decimal_places: int,
                    start_or_end: str
            ) -> pd.DataFrame:
                """
                In an earlier version of the project, I ran into memory issues for two reasons:
                    1) I was dealing with more than a year's worth of data. 
                    2) mistakes that were deeply embedded in the feature pipeline.

                As a result, I decided to match approximations of each coordinate with an ID of 
                my own making, thereby reducing the size of the dataset during the aggregation
                stages of the creation of the training data. This worked well. However, I am no
                longer in need of any memory conservation measures because I have reduced the size of 
                the dataset.

                So why is the code still being used? Why not simply use the original IDs? You 
                may be thinking that perchance there weren't even many missing values, and that 
                perhaps all this could have been avoided.

                There's code in what immediately follows that checks for the presence of both long
                string indices (see the very first method of the class) and missing ones. If the proportion
                of rows that feature such indices exceeds a certain hardcoded threshold (I chose 50%),
                we will use the custom procedures (again see the aforementioned class method).

                As of late July 2024, 60% of the IDs (for origin and destination stations) have long strings
                or missing indices. It is therefore unlikely that we will need an alternative method. However, 
                I will write one eventually. Most likely it will involve simply applying the custom procedure to only
                that problematic minority of indices,to generate new integer indices that aren't already in the column.

                Args:
                    cleaned_data (pd.DataFrame): the version of the dataset that has been cleaned

                    decimal_places (int): the number of decimal places to which we may round the coordinates.
                                          Choosing 6 decimal places will result in no rounding at all.
                    
                    start_or_end (str): whether we are looking at arrivals or departures.

                Returns:
                    pd.DataFrame: the data after the inclusion of the possibly rounded coordinates.
                """
                logger.info(f"Recording the hour during which each trip {start_or_end}s...")
                cleaned_data.insert(
                    loc=cleaned_data.shape[1],
                    column=f"{start_or_end}_hour",
                    value=cleaned_data.loc[:, f"{start_or_end}ed_at"].dt.floor("h"),
                    allow_duplicates=False
                )

                cleaned_data = cleaned_data.drop(f"{start_or_end}ed_at", axis=1)
                logger.info("Determining the method of dealing with invalid station indices...")

                if self.use_custom_station_indexing(scenarios=[start_or_end], data=cleaned_data) and \
                        self.tie_ids_to_unique_coordinates(data=cleaned_data):

                    logger.success("Custom station indexer required: tying new station IDs to unique coordinates")

                    cleaned_data = cleaned_data.drop(columns=f"{start_or_end}_station_id", axis=1)
                    logger.info(f"Approximating the coordinates of the location where each trip {start_or_end}s...")

                    self.indexer = RoundingCoordinates(
                        scenario=start_or_end, data=cleaned_data, decimal_places=decimal_places
                    )

                    # Round the coordinates down to the specified dp, and add these rounded coordinates to the data 
                    self.indexer.add_column_of_rounded_coordinates_to_dataframe(data=cleaned_data)

                    interim_dataframes.append(cleaned_data)
                    logger.info("Matching up approximate locations with generated IDs...")

                    # Make a list of dictionaries of start points and IDs
                    origins_or_destinations_and_ids = self.indexer.make_station_ids_from_unique_coordinates()
                    dictionaries.append(origins_or_destinations_and_ids)

                    # Critical for recovering the (rounded) coordinates and their corresponding IDs later.
                    self.indexer.save_geodata_dict(
                        points_and_ids=origins_or_destinations_and_ids,
                        folder=INDEXER_ONE,
                        file_name=f"rounded_{start_or_end}_points_and_new_ids"
                    )

                    logger.success(f"Done creating IDs for the approximate locations ({start_or_end}s of the trips)")
                    return cleaned_data

                elif self.use_custom_station_indexing(scenarios=[start_or_end], data=cleaned_data) and \
                        not self.tie_ids_to_unique_coordinates(data=cleaned_data):

                    logger.success("Custom station indexer required: NOT tying new IDs to unique coordinates")
                    self.indexer = DirectIndexing(scenario=start_or_end, data=cleaned_data)

                    cleaned_data = self.indexer.full_reindexing(delete_leftover_rows=True)
                    return cleaned_data

                else:
                    raise NotImplementedError(
                        "The majority of Divvy's IDs weren't numerical and valid during initial development."
                    )

            def __aggregate_final_ts(
                    interim_data: pd.DataFrame,
                    start_or_end: str
            ) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:

                if isinstance(self.indexer, RoundingCoordinates):

                    if missing_scenario == "start" or "end":
                        self.indexer.add_column_of_ids(
                            data=interim_data,
                            scenario=start_or_end,
                            points_and_ids=dictionaries[0]
                        )

                    elif missing_scenario == "both":
                        if start_or_end == "start":
                            self.indexer.add_column_of_ids(
                                data=interim_data,
                                scenario=start_or_end,
                                points_and_ids=dictionaries[0]
                            )

                        elif start_or_end == "end":
                            self.indexer.add_column_of_ids(
                                data=interim_data,
                                scenario=start_or_end,
                                points_and_ids=dictionaries[1]
                            )

                    interim_data = interim_data.drop(f"rounded_{start_or_end}_points", axis=1)
                    logger.info(f"Aggregating the final time series data for the {start_or_end}s of trips...")

                agg_data = interim_data.groupby(
                    [f"{start_or_end}_hour", f"{start_or_end}_station_id"]).size().reset_index()

                agg_data = agg_data.rename(columns={0: "trips"})
                return agg_data

            if missing_scenario == "both":
                for data, scenario in zip([start_df, end_df], self.scenarios):
                    # The coordinates are in 6 dp, so no rounding is happening here.
                    __investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario, decimal_places=6)

                # Get all the coordinates that are common to both dictionaries
                common_points = [point for point in dictionaries[0].keys() if point in dictionaries[1].keys()]

                # Ensure that these common points have the same IDs in each dictionary.
                for point in common_points:
                    dictionaries[0][point] = dictionaries[1][point]

                start_ts = __aggregate_final_ts(
                    interim_data=interim_dataframes[0], start_or_end="start"
                )

                end_ts = __aggregate_final_ts(
                    interim_data=interim_dataframes[1], start_or_end="end"
                )

                if save:
                    start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                    end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                return start_ts, end_ts

            elif missing_scenario == "start" or "end":
                data = start_df if missing_scenario == "start" else end_df
                data = __investigate_making_new_station_ids(
                    cleaned_data=data,
                    start_or_end=missing_scenario,
                    decimal_places=5
                )

                ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}s_ts.parquet")

                return ts_data

        return _get_ts_or_begin_transformation(start_ts_path=self.start_ts_path, end_ts_path=self.end_ts_path)

    @staticmethod
    def get_cutoff_indices(ts_data: pd.DataFrame, input_seq_len: int, step_size: int) -> list:
        """
        Starts by taking a certain number of rows of a given dataframe as an input, and the
        indices of the row on which the selected rows start and end. These will be placed
        in the first and second positions of a three element tuple. The third position of
        said tuple will be occupied by the index of the row that comes after.

        Then the function will slide "step_size" steps and repeat the process. The function
        terminates once it reaches the last row of the dataframe.

        Credit to Pau Labarta Bajo.

        Args:
            ts_data (pd.DataFrame): the time series dataset that serves as the input
            
            input_seq_len (int): the number of rows to be considered at any one time

            step_size (int): how many rows down we move as we repeat the process

        Returns:
            list: the list of cutoff indices
        """
        # The function has to stop at the last row of the dataframe
        stop_position = len(ts_data) - 1

        # These numbers will be the first, second, and third elements of each tuple of indices.
        subsequence_first_index = 0
        subsequence_mid_index = input_seq_len
        subsequence_last_index = input_seq_len + 1

        indices = []

        while subsequence_last_index <= stop_position:
            indices.append(
                (subsequence_first_index, subsequence_mid_index, subsequence_last_index)
            )

            subsequence_first_index += step_size
            subsequence_mid_index += step_size
            subsequence_last_index += step_size

        return indices

    def transform_ts_into_training_data(
            self,
            geocode: bool,
            scenario: str,
            step_size: int,
            for_inference: bool,
            input_seq_len: int,
            ts_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transpose the time series data into a feature-target format.

        Args:
            scenario: a string that indicates whether we are dealing with the starts or ends of trips
            
            geocode:
            
            step_size:from src.setup.config import config

            
            input_seq_len:

            ts_data: the time series data

            for_inference (bool): whether we are generating this data as part of the inference pipeline, 
                            or feature pipeline.


        Returns:
            pd.DataFrame: the training data
        """
        if for_inference:
            ts_data = ts_data.drop("timestamp", axis=1)

        # Ensure first that these are the columns of the chosen data set (and they are listed in this order)
        assert set(ts_data.columns) == {f"{scenario}_hour", f"{scenario}_station_id", "trips"}
        station_ids = ts_data[f"{scenario}_station_id"].unique()

        # Prepare the dataframe which will contain the features and targets
        features = pd.DataFrame()
        targets = pd.DataFrame()

        for station_id in tqdm(station_ids):

            # Isolate a part of the dataframe that relates to each station ID
            ts_data_per_station = ts_data.loc[
                ts_data[f"{scenario}_station_id"] == station_id, [f"{scenario}_hour", "trips"]
            ].sort_values(by=[f"{scenario}_hour"])

            # Compute cutoff indices
            indices = self.get_cutoff_indices(
                ts_data=ts_data_per_station, input_seq_len=input_seq_len, step_size=step_size
            )

            num_indices = len(indices)
            # Create a multidimensional array for the features, and a column vector for the target
            x = np.ndarray(shape=(num_indices, input_seq_len), dtype=np.float32)
            y = np.ndarray(shape=num_indices, dtype=np.float32)

            hours = []
            for i, index in enumerate(indices):
                x[i, :] = ts_data_per_station.iloc[index[0]: index[1]]["trips"].values
                y[i] = ts_data_per_station[index[1]:index[2]]["trips"].values[0]

                # Append the "hours" list with the appropriate entry at the intersection
                # of row "index[1]" and scenario_hour's column
                hours.append(
                    ts_data_per_station.iloc[index[1]][f"{scenario}_hour"]
                )

            # Make a dataframe of features
            features_per_location = pd.DataFrame(
                x, columns=[
                    f"trips_previous_{i + 1}_hour" for i in reversed(range(input_seq_len))
                ]
            )

            features_per_location[f"{scenario}_hour"] = hours
            features_per_location[f"{scenario}_station_id"] = station_id

            targets_per_location = pd.DataFrame(
                y, columns=["trips_next_hour"]
            )

            with warnings.catch_warnings():

                # There is a warning from pandas about concatenating dataframes that are empty or have missing values
                warnings.filterwarnings(action="ignore", category=FutureWarning)

                # Concatenate the dataframes
                features = pd.concat([features, features_per_location], axis=0)
                targets = pd.concat([targets, targets_per_location], axis=0)

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        engineered_features = perform_feature_engineering(features=features, scenario=scenario, geocode=geocode)

        if not for_inference:
            data_to_save = pd.concat([engineered_features, targets["trips_next_hour"]], axis=1)
        else:
            data_to_save = engineered_features

        logger.success("Saving the data so we (hopefully) won't have to do that again...")

        final_data_path = INFERENCE_DATA if for_inference else TRAINING_DATA
        data_to_save.to_parquet(path=final_data_path / f"{scenario}s.parquet")
        return data_to_save


if __name__ == "__main__":
    make_fundamental_paths()
    trips_2024 = DataProcessor(year=2024, for_inference=False)
    trips_2024.make_training_data(geocode=False)
