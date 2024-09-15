import json
import warnings
from tqdm import tqdm
from loguru import logger
from pathlib import Path

import numpy as np
import pandas as pd

from src.setup.config import config
from src.feature_pipeline.data_extraction import load_raw_data
from src.feature_pipeline.station_indexing import RoundingCoordinates, DirectIndexing
from src.feature_pipeline.feature_engineering import perform_feature_engineering

from src.setup.paths import (
    CLEANED_DATA, TRAINING_DATA, RAW_DATA_DIR, TIME_SERIES_DATA, INDEXER_TWO, INFERENCE_DATA, make_fundamental_paths, PARQUETS
)


class DataProcessor:
    def __init__(self, year: int, for_inference: bool):
        self.station_ids = None
        self.scenarios = ["start", "end"]
        self.for_inference = for_inference
        self.start_ts_path = TIME_SERIES_DATA / "start_ts.parquet"
        self.end_ts_path = TIME_SERIES_DATA / "end_ts.parquet"

        if for_inference:
            self.data = None  # Because the data will have been fetched from the feature store instead.
        else:
            loaded_raw_data = list(load_raw_data(year=year))
            self.data = pd.concat(loaded_raw_data) 

    def use_custom_station_indexing(self, scenarios: list[str], data: pd.DataFrame) -> bool:
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
        if not self.for_inference:
            results = []

            for scenario in scenarios:
                long_id_count = 0

                for station_id in data.loc[:, f"{scenario}_station_id"]:
                    if len(str(station_id)) >= 7 and not pd.isnull(station_id):
                        long_id_count += 1

                number_of_missing_indices = data[f"{scenario}_station_id"].isna().sum()
                proportion_of_problem_rows = (number_of_missing_indices + long_id_count) / data.shape[0] 
                result = True if proportion_of_problem_rows >= 0.5 else False

                results.append(result)

            return True if False not in results else False

    def tie_ids_to_unique_coordinates(self, data: pd.DataFrame) -> bool:
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
        if not self.for_inference:
            return True if len(data) > 10_000_000 else False

    def make_training_data(self, geocode: bool) -> list[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data, clean it, transform it into time series data, and transform that in turn into
        training data which is subsequently saved. 
-
        Args:
            geocode (bool): whether to geocode as part of feature engineering.
            
        Returns:
            list[pd.DataFrame]: a list containing the datasets for the starts and ends of trips.
        """
        start_ts, end_ts = self.make_time_series()
        ts_data_per_scenario = {"start": start_ts, "end": end_ts}

        training_sets = []
        for scenario in ts_data_per_scenario.keys():

            if not Path(TRAINING_DATA/f"{scenario}s.parquet").is_file():

                training_data = self.transform_ts_into_training_data(
                    ts_data=ts_data_per_scenario[scenario],
                    geocode=geocode,
                    scenario=scenario,
                    input_seq_len=1,
                    step_size=1,
                    track_partials=True
                )

                training_sets.append(training_data)
            else:
                logger.success(f"You already have training data for the {config.displayed_scenario_names[scenario]}s")  
                continue            

            return training_sets

    def make_time_series(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform the transformation of the raw data into time series data 

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets for departures and arrivals respectively.
        """
        logger.info("Cleaning downloaded data...")
        self.data = self.clean()

        start_df_columns = ["started_at", "start_lat", "start_lng", "start_station_id"]
        end_df_columns = ["ended_at", "end_lat", "end_lng", "end_station_id"]

        if self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) and not \
                self.tie_ids_to_unique_coordinates(data=self.data):

            start_df_columns.append("start_station_name")
            end_df_columns.append("end_station_name")

        start_df = self.data[start_df_columns]
        end_df = self.data[end_df_columns]

        logger.info("Transforming data into a time series...")
        start_ts, end_ts = self.transform_cleaned_data_into_ts_data(start_df=start_df, end_df=end_df)
        return start_ts, end_ts

    def clean(self, save: bool = True) -> pd.DataFrame:

        if self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) \
                and self.tie_ids_to_unique_coordinates(data=self.data):

            cleaned_data_file_path = CLEANED_DATA / "data_with_newly_indexed_stations (indexer_one).parquet"

        elif self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data) \
                and not self.tie_ids_to_unique_coordinates(data=self.data):

            cleaned_data_file_path = CLEANED_DATA/"data_with_newly_indexed_stations (indexer_two).parquet"

        # Will think of a more elegant solution in due course. This only serves my current interests.
        elif self.for_inference:
            cleaned_data_file_path = CLEANED_DATA / "partially_cleaned_data_for_inference.parquet"

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

                    logger.info(f"Deleting rows with missing station names and coordinates ({config.displayed_scenario_names[scenario]})...")

                    where_missing_latitudes = self.data.iloc[:, lats].isnull()
                    where_missing_longitudes = self.data.iloc[:, longs].isnull()
                    where_missing_station_names = self.data.iloc[:, station_names_col].isnull()

                    all_missing_mask = where_missing_station_names & where_missing_latitudes & where_missing_longitudes
                    data_to_delete = self.data.loc[all_missing_mask, :]

                    self.data = self.data.drop(data_to_delete.index, axis=0)

                return self.data

            self.data = _delete_rows_with_missing_station_names_and_coordinates()
            features_to_drop = ["ride_id", "rideable_type", "member_casual"]

            if self.use_custom_station_indexing(data=self.data, scenarios=self.scenarios) and \
                    self.tie_ids_to_unique_coordinates(data=self.data):
                features_to_drop.extend(
                    ["start_station_id", "start_station_name", "end_station_name"]
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

        src.feature_pipeline.In addition to the putting the arrival and departure times in hourly form, we approximate
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
                logger.warning("No time series dataset for departures has been made")
                start_ts = _begin_transformation(missing_scenario="start")
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("No time series dataset for arrivals has been made")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = _begin_transformation(missing_scenario="end")
                return start_ts, end_ts

        def _begin_transformation(missing_scenario: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:

            interim_dataframes: list[pd.DataFrame] = []

            def __investigate_making_new_station_ids(cleaned_data: pd.DataFrame, start_or_end: str) -> pd.DataFrame:
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
                    start_or_end (str): whether we are looking at arrivals or departures.

                Returns:
                    pd.DataFrame:
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

                if self.use_custom_station_indexing(scenarios=[start_or_end], data=self.data) and \
                        self.tie_ids_to_unique_coordinates(data=self.data):

                    logger.warning("Custom station indexer required: tying new station IDs to unique coordinates")
                    indexer = RoundingCoordinates(data=cleaned_data, scenario=start_or_end, decimal_places=4)

                    interim_data = indexer.execute()
                    interim_dataframes.append(interim_data)

                    return interim_data

                elif self.use_custom_station_indexing(scenarios=[start_or_end], data=self.data) and \
                        not self.tie_ids_to_unique_coordinates(data=self.data):

                    logger.warning("Custom station indexer required: NOT tying new IDs to unique coordinates")
        
                    indexer = DirectIndexing(scenario=start_or_end, data=cleaned_data)
                    interim_data = indexer.execute(delete_leftover_rows=True)
                    interim_dataframes.append(interim_data)
                    return interim_data

                else:
                    raise NotImplementedError(
                        "The majority of Divvy's IDs weren't numerical and valid during initial development."
                    )

            def __aggregate_final_ts(
                interim_data: pd.DataFrame,
                start_or_end: str
            ) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:

                #  if self.use_custom_station_indexing(data=self.data, scenarios=[start_or_end]) and \
                #        self.tie_ids_to_unique_coordinates(data=self.data):

                #    interim_data = interim_data.drop(f"rounded_{start_or_end}_points", axis=1)

                logger.info(f"Aggregating the final time series data for the {config.displayed_scenario_names[start_or_end].lower()}...")

                agg_data = interim_data.groupby(
                    [f"{start_or_end}_hour", f"{start_or_end}_station_id"]).size().reset_index()

                agg_data = agg_data.rename(columns={0: "trips"})
                return agg_data

            indexer_two_scenarios = self.scenarios if missing_scenario == "both" else [missing_scenario]

            if self.use_custom_station_indexing(scenarios=indexer_two_scenarios, data=self.data) and \
                    self.tie_ids_to_unique_coordinates(data=self.data):

                if missing_scenario == "both":

                    for data, scenario in zip([start_df, end_df], self.scenarios):
                        # The coordinates are in 6 dp, so no rounding is happening here.
                        __investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                    with open(INDEXER_TWO/"rounded_start_points_and_new_ids.json", mode="r") as file:
                        rounded_start_points_and_ids = json.load(file)

                    with open(INDEXER_TWO/"rounded_end_points_and_new_ids.json", mode="r") as file:
                        rounded_end_points_and_ids = json.load(file)

                    # Get all the coordinates that are common to both dictionaries
                    common_points = [
                        point for point in rounded_start_points_and_ids.keys() if point in
                        rounded_end_points_and_ids.keys()
                    ]

                    # Ensure that these common points have the same IDs in each dictionary.
                    for point in common_points:
                        rounded_start_points_and_ids[point] = rounded_end_points_and_ids[point]

                    start_ts = __aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                    end_ts = __aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                    if save:
                        start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                        end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                    return start_ts, end_ts

                elif missing_scenario == "start" or "end":
                    
                    data = start_df if missing_scenario == "start" else end_df
                    data = __investigate_making_new_station_ids(cleaned_data=data, start_or_end=missing_scenario)
                    ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                    if save:
                        ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}s_ts.parquet")

                    return ts_data

            elif self.use_custom_station_indexing(scenarios=indexer_two_scenarios, data=self.data) and \
                    not self.tie_ids_to_unique_coordinates(data=self.data):

                if missing_scenario == "both":
                    for data, scenario in zip([start_df, end_df], self.scenarios):
                        __investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                    start_ts = __aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                    end_ts = __aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                    if save:
                        start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                        end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                    return start_ts, end_ts

                elif missing_scenario == "start" or "end":

                    data = start_df if missing_scenario == "start" else end_df
                    data = __investigate_making_new_station_ids(cleaned_data=data, start_or_end=missing_scenario)
                    ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                    if save:
                        ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}_ts.parquet")

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

        Credit to P.L.B.

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
            input_seq_len: int,
            ts_data: pd.DataFrame,
            track_partials: bool
    ) -> pd.DataFrame:
        """
        Transpose the time series data into a feature-target format.

        Args:
            scenario: a string that indicates whether we are dealing with the starts or ends of trips
            
            geocode:
            
            step_size:

            
            input_seq_len:

            ts_data: the time series data

        Returns:
            pd.DataFrame: the training data
        """
        if self.for_inference and "timestamp" in ts_data.columns:
            ts_data = ts_data.drop("timestamp", axis=1)

        # Ensure first that these are the columns of the chosen data set (and they are listed in this order)
        assert set(ts_data.columns) == {f"{scenario}_hour", f"{scenario}_station_id", "trips"}

        # Prepare the dataframe which will contain the features and targets
        features = pd.DataFrame()
        targets = pd.DataFrame()

        for station_id in tqdm(
            iterable=ts_data[f"{scenario}_station_id"].unique(), 
            desc=f"Turning time series data into training data ({config.displayed_scenario_names[scenario].lower()})"
        ):
            
            # Isolate a part of the dataframe that relates to each station ID
            ts_data_per_station = ts_data.loc[
                ts_data[f"{scenario}_station_id"] == station_id, [f"{scenario}_hour", f"{scenario}_station_id", "trips"]
            ].sort_values(by=[f"{scenario}_hour"])

            # Compute cutoff indices
            indices = self.get_cutoff_indices(
                ts_data=ts_data_per_station, 
                input_seq_len=input_seq_len, 
                step_size=step_size
            )

            # Create a multidimensional array for the features, and a column vector for the target
            num_indices = len(indices)
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
            features_per_station = pd.DataFrame(
                x, columns=[
                    f"trips_previous_{i + 1}_hour" for i in reversed(range(input_seq_len))
                ]
            )
            
            if track_partials:
                features_per_station.to_csv(PARQUETS/f"#{station_id}.csv")

            features_per_station[f"{scenario}_hour"] = hours
            features_per_station[f"{scenario}_station_id"] = station_id

            #features_per_station.to_parquet(path=PARQUETS/f"#{station_id}.parquet")

            targets_per_location = pd.DataFrame(
                y, columns=["trips_next_hour"]
            )

            # Concatenate the dataframes
            features = pd.concat([features, features_per_station], axis=0)
            targets = pd.concat([targets, targets_per_location], axis=0)

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)

        features.to_parquet(TRAINING_DATA/f"{scenario}_raw_features.parquet")
        breakpoint()

        engineered_features = perform_feature_engineering(features=features, scenario=scenario, geocode=geocode)
        training_data = pd.concat([engineered_features, targets["trips_next_hour"]], axis=1)

        logger.success("Saving the data so we (hopefully) won't have to do that again...")
        final_data_path = INFERENCE_DATA if self.for_inference else TRAINING_DATA

        if scenario == "end":
            training_data.to_csv(final_data_path / f"{scenario}s.csv")
        return training_data


if __name__ == "__main__":
    make_fundamental_paths()
    trips_2024 = DataProcessor(year=2024, for_inference=False)
    trips_2024.make_training_data(geocode=False)
