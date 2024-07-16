import os
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from pathlib import Path

from src.setup.miscellaneous import (
    add_column_of_rounded_points, add_column_of_ids, make_dict_of_new_station_ids,
    add_rounded_coordinates_to_dataframe, save_geodata_dict
)

from src.feature_pipeline.data_extraction import load_raw_data
from src.feature_pipeline.feature_engineering import perform_feature_engineering
from src.setup.paths import CLEANED_DATA, TRAINING_DATA, TIME_SERIES_DATA, GEOGRAPHICAL_DATA, make_fundamental_paths


class DataProcessor:
    def __init__(self, year: int):
        self.data = pd.concat(
            list(load_raw_data(year=year))
        )
        self.station_ids = None
        self.starts_ts_path = TIME_SERIES_DATA/"starts_ts.parquet"
        self.ends_ts_path = TIME_SERIES_DATA/"ends_ts.parquet"

    def make_training_data(
            self,
            for_feature_store: bool,
            geocode: bool
    ) -> list[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data, transform it into a time series, and transform that time series into
        training data which is subsequently saved.

        Args:
            for_feature_store (bool): whether we are intending to stop at the provision of
                                      time series data for subsequent pushing to the feature
                                      store.
            geocode (bool): whether to geocode when performing feature engineering.

        Returns:
            list[pd.DataFrame]: a list containing the datasets for the starts and ends of trips.
        """
        logger.info("Cleaning dataframe")
        self.data = self.clean()

        starts = self.data[
            ["start_time", "start_lat", "start_lng", "start_station_id"]
        ]

        ends = self.data[
            ["end_time", "end_lat", "end_lng", "end_station_id"]
        ]

        logger.info("Transforming the data into a time series...")
        starts_ts, ends_ts = self.transform_cleaned_data_into_ts_data(start_df=starts, end_df=ends)
        if for_feature_store:
            return starts_ts, ends_ts

        ts_data_per_scenario = {"start": starts_ts, "end": ends_ts}

        training_sets = []
        for scenario in ts_data_per_scenario.keys():
            logger.info(f"Turning the time series data on the {scenario}s of trips into training data...")
            training_data = self.transform_ts_into_training_data(
                ts_data=ts_data_per_scenario[scenario],
                geocode=geocode,
                scenario=scenario,
                input_seq_len=24 * 28 * 1,
                step_size=24
            )

            logger.success("Saving the data so that we don't have to do this again...")
            training_data.to_parquet(path=TRAINING_DATA/f"{scenario}s.parquet")
            training_sets.append(training_data)

        return training_sets

    def clean(self, patient: bool = False, save: bool = True) -> pd.DataFrame:

        if len(os.listdir(path=CLEANED_DATA)) == 0:
            self.data["started_at"] = pd.to_datetime(self.data["started_at"], format="mixed")
            self.data["ended_at"] = pd.to_datetime(self.data["ended_at"], format="mixed")

            self.data = self.data.rename(
                columns={
                    "started_at": "start_time",
                    "ended_at": "end_time"
                }
            )

            def _delete_rows_with_missing_station_names_and_coordinates() -> pd.DataFrame:
                """
                There are rows with missing latitude and longitude values for the various
                destinations. If any of these rows have available station names, then geocoding
                can be used to get the coordinates. At the current time however, all rows with
                missing coordinates also have missing station names, rendering those rows
                irreparably lacking. 
                
                We locate and delete these points with this function.

                Returns:
                    pd.DataFrame: the data, absent the aforementioned rows.
                """
                for scenario in ["start", "end"]:
                    lats = self.data.columns.get_loc(f"{scenario}_lat")
                    longs = self.data.columns.get_loc(f"{scenario}_lng")
                    station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                    all_rows = tqdm(
                        iterable=range(self.data.shape[0]),
                        desc=f"Targeting rows with missing station names and coordinates for deletion ({scenario}s)"
                    )

                    rows = []
                    for row in all_rows:
                        if (
                                pd.isnull(self.data.iloc[row, station_names_col])
                                and pd.isnull(self.data.iloc[row, lats]) and pd.isnull(self.data.iloc[row, longs])
                        ):
                            rows.append(row)

                    # Check that all rows with missing latitudes and longitudes also have missing station names
                    assert len(rows) == self.data.isna().sum()[f"{scenario}_lat"] == self.data.isna().sum()[
                        f"{scenario}_lng"]
                    self.data = self.data.drop(self.data.index[rows], axis=0)

                return self.data

            def _find_rows_with_missing_station_names_and_ids(scenario: str) -> list:
                station_id_col = self.data.columns.get_loc(f"{scenario}_station_id")
                station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                all_rows = tqdm(
                    iterable=range(self.data.shape[0]),
                    desc="Searching for rows with missing station names and IDs"
                )

                sought_rows = []
                for row in all_rows:
                    if pd.isnull(self.data.iloc[row, station_names_col]) and \
                            pd.isnull(self.data.iloc[row, station_id_col]):
                        sought_rows.append(row)
                return sought_rows

            def _find_rows_with_known_coordinates_names_and_ids(scenario: str) -> tuple[list, list, list, list]:

                lats = self.data.columns.get_loc(f"{scenario}_lat")
                longs = self.data.columns.get_loc(f"{scenario}_lng")
                station_id_col = self.data.columns.get_loc(f"{scenario}_station_id")
                station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                for _ in tqdm(range(self.data.shape[0])):
                    sought_rows = []
                    rows_with_known_lats = []
                    rows_with_known_longs = []
                    rows_with_known_ids = []
                    rows_with_known_station_names = []

                    all_rows = tqdm(
                        iterable=range(self.data.shape[0]),
                        desc="Finding rows with known coordinates, station names and IDs"
                    )

                    for row in all_rows:
                        if (
                                not pd.isnull(self.data.iloc[row, lats]) and not pd.isnull(self.data.iloc[row, longs])
                                and not pd.isnull(self.data.iloc[row, station_id_col])
                                and not pd.isnull(self.data.iloc[row, station_names_col])
                        ):

                            sought_rows.append(row)
                            rows_with_known_lats.append(self.data.iloc[row, lats])
                            rows_with_known_longs.append(self.data.iloc[row, longs])
                            rows_with_known_ids.append(self.data.iloc[row, station_id_col])
                            rows_with_known_station_names.append(self.data.iloc[row, station_names_col])

                    return (
                        rows_with_known_lats,
                        rows_with_known_longs,
                        rows_with_known_ids,
                        rows_with_known_station_names
                    )

            def _replace_missing_names_and_ids(
                    scenario: str,
                    rows_with_known_lats: list,
                    rows_with_known_longs: list,
                    rows_with_known_station_ids: list,
                    rows_with_known_station_names: list
            ) -> pd.DataFrame:

                rows_to_search = tqdm(
                    iterable=rows_missing_station_names_ids,
                    desc="Searching through rows to find matching latitudes and longitudes"
                )
                lats = self.data.columns.get_loc(f"{scenario}_lat")
                longs = self.data.columns.get_loc(f"{scenario}_lng")
                station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")
                station_id_col = self.data.columns.get_loc(f"{scenario}_station_id")

                for row in rows_to_search:
                    for lat, lng in zip(rows_with_known_lats, rows_with_known_longs):
                        if lat == self.data.iloc[row, lats] and lng == self.data.iloc[row, longs]:
                            self.data = self.data.replace(
                                to_replace=self.data.iloc[row, station_id_col],
                                value=rows_with_known_station_ids[rows_with_known_lats.index(lat)]
                            )

                            self.data = self.data.replace(
                                to_replace=self.data.iloc[row, station_names_col],
                                value=rows_with_known_station_names[rows_with_known_lats.index(lat)]
                            )
                return self.data

            self.data = _delete_rows_with_missing_station_names_and_coordinates()

            if patient:
                rows_missing_station_names_ids = _find_rows_with_missing_station_names_and_ids(scenario="end")
                known_lats, known_longs, known_station_ids, known_station_names = \
                    _find_rows_with_known_coordinates_names_and_ids(scenario="end")

                self.data = _replace_missing_names_and_ids(
                    scenario="end",
                    rows_with_known_lats=known_lats,
                    rows_with_known_longs=known_longs,
                    rows_with_known_station_ids=known_station_ids,
                    rows_with_known_station_names=known_station_names
                )

                self.data = self.data.drop(
                    columns=[
                        "ride_id", "rideable_type", "member_casual"
                    ]
                )

            else:
                self.data = self.data.drop(
                    columns=[
                        "ride_id", "rideable_type", "member_casual", "start_station_name", "end_station_name"
                    ]
                )

            if save:
                self.data.to_parquet(path=CLEANED_DATA / "cleaned.parquet")

            return self.data

        else:
            logger.success("There is already some cleaned data. Fetching it...")
            return pd.read_parquet(path=CLEANED_DATA / "cleaned.parquet")

    def transform_cleaned_data_into_ts_data(
            self,
            start_df: pd.DataFrame,
            end_df: pd.DataFrame,
            save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts cleaned data into time series data.

        In addition to the putting the start and end times in hourly form, we approximate
        the latitudes and longitudes of each point of origin or destination (we are targeting
        no more than a 100m radius of each point, but ideally we would like to maintain a 10m
        radius), and use these to construct new station IDs.

        Args:
            start_df (pd.DataFrame): dataframe consisting of the "start_time", "start_lat",
                                     and "start_lng" columns.

            end_df (pd.DataFrame): dataframe consisting of the "stop_time", "stop_latitude",
                                   and "stop_lng" columns.

            save (bool): whether we wish to save the time series data

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on the starts and ends
                                               of trips.
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
                logger.warning("Time series data for trip starts has not been made")
                start_ts = _begin_transformation(missing_scenario="start")
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("Time series data for the ends of trips has not been made")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = _begin_transformation(missing_scenario="end")
                return start_ts, end_ts

        def _begin_transformation(missing_scenario: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:

            dictionaries: list[dict] = []
            interim_dataframes: list[pd.DataFrame] = []

            def __round_coordinates_and_make_ids(
                    cleaned_data: pd.DataFrame,
                    decimal_places: int,
                    start_or_end: str
            ) -> pd.DataFrame:
                """
                In an earlier version of the project, I ran into memory issues for two reasons:
                    1) I was dealing with more than a year's worth of data. 
                    2) Due a mistake that was deeply embedded in the feature pipeline of the
                       project.

                As a result, I decided to match approximations of each coordinate with an ID of 
                my own making. thereby reducing the size of the dataset during the aggregation 
                stages of the creation of the training data. This worked well. However, I am no
                longer in need of any memory conservation measures.

                So why is the code still being used? Why not simply use the original IDs? You 
                may be thinking that perchance there weren't even many missing values. That 
                perhaps all this could have been avoided.

                There's code in what immediately follows that checks for the presence of both
                string indices (which are somewhat problematic) and missing ones. If the proportion
                of rows that feature such indices exceeds a certain hardcoded threshold (I chose 50%),
                we use the custom procedure described above (without rounding.ie. with the original
                coordinates). As of early July 2024, and for the IDs of origin and destination stations, 
                82% of the rows have string or missing indices. It is therefore unlikely that we will need 
                an alternative method. However, I will write one eventually. Most likely it will involve
                simply applying the custom procedure to only that problematic minority of indices,to generate
                new integer indices that aren't already in the column.

                Args:
                    cleaned_data (pd.DataFrame): the version of the dataset that has been cleaned
                    decimal_places (int): the number of decimal places to which we may round the coordinates.
                                          Choosing 6 decimal places will result in no rounding at all.
                    start_or_end (str): whether we are looking at the starts or ends of trips.

                Returns:
                    pd.DataFrame: the data after the inclusion of the possibly rounded coordinates.
                """

                def find_num_string_indices() -> int:
                    string_count = 0
                    for station_id in cleaned_data[f"{start_or_end}_station_id"].to_list():
                        if isinstance(station_id, str):
                            string_count += 1
                    return string_count

                def use_custom_indexing_method() -> bool:
                    string_count = find_num_string_indices()
                    num_missing_indices = cleaned_data[f"{start_or_end}_station_id"].isna().sum()
                    if (num_missing_indices + string_count) / cleaned_data.shape[0] > 0.5:
                        return True
                    else:
                        return False

                if use_custom_indexing_method():
                    cleaned_data = cleaned_data.drop(f"{start_or_end}_station_id", axis=1)
                    logger.info(f"Recording the hour during which each trip {start_or_end}s...")

                    cleaned_data.insert(
                        loc=cleaned_data.shape[1],
                        column=f"{start_or_end}_hour",
                        value=cleaned_data.loc[:, f"{start_or_end}_time"].dt.floor("h"),
                        allow_duplicates=False
                    )

                    cleaned_data = cleaned_data.drop(f"{start_or_end}_time", axis=1)
                    logger.info(f"Approximating the coordinates of the location where each trip {start_or_end}s...")
                    # Round the lats and longs down to the specified dp, and add the rounded values to the data
                    add_rounded_coordinates_to_dataframe(
                        data=cleaned_data,
                        decimal_places=decimal_places,
                        scenario=start_or_end
                    )

                    cleaned_data.drop(
                        columns=[f"{start_or_end}_lat", f"{start_or_end}_lng"],
                        inplace=True
                    )

                    # Add the rounded coordinates to the dataframe as a column.
                    add_column_of_rounded_points(data=cleaned_data, scenario=start_or_end)

                    cleaned_data.drop(
                        columns=[f"rounded_{start_or_end}_lat", f"rounded_{start_or_end}_lng"],
                        inplace=True
                    )

                    interim_dataframes.append(cleaned_data)
                    logger.info("Matching up approximate locations with generated IDs...")

                    # Make a list of dictionaries of start points and IDs
                    origins_or_destinations_and_ids = make_dict_of_new_station_ids(
                        data=cleaned_data,
                        scenario=start_or_end
                    )
                    dictionaries.append(origins_or_destinations_and_ids)

                    # Critical for recovering the (rounded) coordinates and their corresponding IDs later.
                    save_geodata_dict(
                        dictionary=origins_or_destinations_and_ids,
                        folder=GEOGRAPHICAL_DATA,
                        file_name=f"rounded_{start_or_end}_points_and_new_ids"
                    )

                    logger.success(f"Done with the {start_or_end}s of the trips!")
                    return cleaned_data

                else:
                    raise NotImplementedError("Yet to provide logic to perform an alternative indexing method")

            def __aggregate_final_ts(
                    interim_data: pd.DataFrame,
                    start_or_end: str
            ) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:

                if missing_scenario == "start" or "end":
                    add_column_of_ids(data=interim_data, scenario=start_or_end, points_and_ids=dictionaries[0])

                elif missing_scenario == "both":
                    if start_or_end == "start":
                        add_column_of_ids(data=interim_data, scenario=start_or_end, points_and_ids=dictionaries[0])
                    elif start_or_end == "end":
                        add_column_of_ids(data=interim_data, scenario=start_or_end, points_and_ids=dictionaries[1])

                interim_data = interim_data.drop(f"rounded_{start_or_end}_points", axis=1)
                logger.info(f"Aggregating the final time series data for the {start_or_end}s of trips...")

                agg_data = interim_data.groupby(
                    [f"{start_or_end}_hour", f"{start_or_end}_station_id"]).size().reset_index()

                agg_data = agg_data.rename(columns={0: "trips"})
                return agg_data

            if missing_scenario == "both":
                for data, scenario in zip([start_df, end_df], ["start", "end"]):
                    # The coordinates are in 6 dp, so no rounding is happening here.
                    __round_coordinates_and_make_ids(cleaned_data=data, start_or_end=scenario, decimal_places=6)

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
                    start_ts.to_parquet(TIME_SERIES_DATA/"starts_ts.parquet")
                    end_ts.to_parquet(TIME_SERIES_DATA/"ends_ts.parquet")

                return start_ts, end_ts

            elif missing_scenario == "start" or "end":
                data = start_df if missing_scenario == "start" else end_df
                data = __round_coordinates_and_make_ids(
                    cleaned_data=data,
                    start_or_end=missing_scenario,
                    decimal_places=5
                )

                ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}s_ts.parquet")

                return ts_data

        return _get_ts_or_begin_transformation(start_ts_path=self.starts_ts_path, end_ts_path=self.ends_ts_path)

    @staticmethod
    def get_cutoff_indices(ts_data: pd.DataFrame, input_seq_len: int, step_size_len: int) -> list:
        """
        Starts by taking a certain number of rows of a given dataframe as an input, and the
        indices of the row on which the selected rows start and end. These will be placed
        in the first and second positions of a three element tuple. The third position of
        said tuple will be occupied by the index of the row that comes after.

        Then the function will slide "step_size_len" steps and repeat the process. The function
        terminates once it reaches the last row of the dataframe.

        Credit to Pau Labarta Bajo.

        Args:
            ts_data (pd.DataFrame): the time series dataset that serves as the input
            input_seq_len (int): the number of rows to be considered at any one time
            step_size_len (int): how many rows down we move as we repeat the process

        Returns:
            list: the list of cutoff indices
        """
        # The function has to stop at the last row of the dataframe
        stop_position = len(ts_data) - 1

        # These numbers will be the first, second, and third elements of each tuple of indices.
        subseq_first_index = 0
        subseq_mid_index = input_seq_len
        subseq_last_index = input_seq_len + 1

        indices = []

        while subseq_last_index <= stop_position:
            indices.append(
                (subseq_first_index, subseq_mid_index, subseq_last_index)
            )

            subseq_first_index += step_size_len
            subseq_mid_index += step_size_len
            subseq_last_index += step_size_len

        return indices

    def transform_ts_into_training_data(
            self,
            geocode: bool,
            scenario: str,
            step_size: int,
            input_seq_len: int,
            ts_data: pd.DataFrame
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
                ts_data=ts_data_per_station,
                input_seq_len=input_seq_len,
                step_size_len=step_size
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
                # of row "index[1]" and the appropriate scenario's column
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
            targets_per_location = pd.DataFrame(y, columns=["trips_next_hour"])

            with warnings.catch_warnings():
                # There is a warning from pandas about concatenating dataframes that are empty or have missing values
                warnings.filterwarnings(action="ignore", category=FutureWarning)

                # Concatenate the dataframes
                features = pd.concat([features, features_per_location], axis=0)
                targets = pd.concat([targets, targets_per_location], axis=0)

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        engineered_features = perform_feature_engineering(features=features, scenario=scenario, geocode=geocode)
        training_data = pd.concat(
            [engineered_features, targets["trips_next_hour"]], axis=1
        )

        return training_data


if __name__ == "__main__":
    make_fundamental_paths()
    trips_2024 = DataProcessor(year=2024)
    trips_2024.make_training_data(geocode=False, for_feature_store=False)
