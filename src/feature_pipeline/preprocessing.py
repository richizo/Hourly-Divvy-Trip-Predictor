import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from pathlib import Path

from src.setup.miscellaneous import (
    add_column_of_rounded_points, add_column_of_ids, make_dict_of_new_station_ids, add_rounded_coordinates_to_dataframe, save_dict
)

from src.feature_pipeline.data_extraction import load_raw_data
from src.setup.paths import CLEANED_DATA, TRAINING_DATA, TIME_SERIES_DATA, GEOGRAPHICAL_DATA, make_fundamental_paths


class DataProcessor:
    def __init__(self, year: int):

        self.data = pd.concat(
            list(load_raw_data(year=2024))
        )

        self.starts_ts_path = TIME_SERIES_DATA/"starts_ts.parquet"
        self.ends_ts_path = TIME_SERIES_DATA/"ends_ts.parquet"


    def _clean(self, patient: bool = False, save: bool = True) -> pd.DataFrame:

        if len(os.listdir(path=CLEANED_DATA)) == 0:

            self.data["started_at"] = pd.to_datetime(self.data["started_at"], format="mixed")
            self.data["ended_at"] = pd.to_datetime(self.data["ended_at"], format="mixed")

            self.data = self.data.rename(
                columns={"started_at": "start_time", "ended_at": "end_time"}
            )

            def __delete_rows_with_unnamed_and_missing_coordinates() -> pd.DataFrame:
                """
                There are rows with missing latitude and longitude values for the various 
                destinations. If any of these rows have available station names, then geocoding 
                can be used to get the coordinates. At the current time however, all rows with 
                missing coordinates also have missing station names, rendering those rows 
                irreprably lacking. We locate and delete these points.

                Returns:
                    pd.DataFrame: the data, absent the aforementioned rows.
                """ 
                for scenario in ["start", "end"]:
                    station_names = self.data.columns.get_loc(f"{scenario}_station_name")
                    lats = self.data.columns.get_loc(f"{scenario}_lat")
                    lngs = self.data.columns.get_loc(f"{scenario}_lng")

                    all_rows = tqdm(
                        iterable=range(self.data.shape[0]),
                        desc=f"Targeting rows with missing station names and coordinates for deletion ({scenario}s of trips)"
                    )

                    rows = []
                    for row in all_rows:
                        if pd.isnull(self.data.iloc[row, station_names]) and pd.isnull(self.data.iloc[row, lats]) and pd.isnull(
                                self.data.iloc[row, lngs]
                            ):
                            rows.append(row)

                    # Check that all rows with missing latitudes and longitudes also have missing station names
                    assert len(rows) == self.data.isna().sum()[f"{scenario}_lat"] == self.data.isna().sum()[f"{scenario}_lng"]
                    self.data = self.data.drop(self.data.index[rows], axis=0)

                return self.data

            def __find_rows_with_missing_station_names_ids(scenario: str) -> list:
                station_id = self.data.columns.get_loc(f"{scenario}_station_id")
                station_names = self.data.columns.get_loc(f"{scenario}_station_name")

                all_rows = tqdm(
                    iterable=range(self.data.shape[0]),
                    desc="Searching for rows with missing station names and IDs"
                )

                rows = []
                for row in all_rows:
                    if pd.isnull(self.data.iloc[row, station_names]) and pd.isnull(self.data.iloc[row, station_id]):
                        rows.append(row)
                return rows
        
            def _find_rows_with_known_coords_names_and_ids(scenario: str) -> tuple[list, list, list, list]:

                lats = self.data.columns.get_loc(f"{scenario}_lat")
                lngs = self.data.columns.get_loc(f"{scenario}_lng")
                    
                for row in tqdm(range(self.data.shape[0])):
                    
                    station_id_col = self.data.columns.get_loc(f"{scenario}_station_id")
                    station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                    rows = []
                    known_lats = []
                    known_lngs = []
                    known_station_ids = []
                    known_station_names = []
                    
                    all_rows = tqdm(
                        iterable=range(self.data.shape[0]), 
                        desc="Finding rows with known coordinates, station names and IDs"
                    )

                    for row in all_rows:
                        if not pd.isnull(self.data.iloc[row, lats]) and not pd.isnull(self.data.iloc[row, lngs]) and not pd.isnull(
                                self.data.iloc[row, station_id_col]) and not pd.isnull(self.data.iloc[row, station_names_col]):
                            
                            rows.append(row) 
                            known_lats.append(self.data.iloc[row, lats])
                            known_lngs.append(self.data.iloc[row, lngs])
                            known_station_ids.append(self.data.iloc[row, station_id_col])
                            known_station_names.append(self.data.iloc[row, station_names_col])

                    return known_lats, known_lngs, known_station_ids, known_station_names

            def _replace_missing_names_and_ids(
                scenario: str,
                known_lats: list, 
                known_lngs: list, 
                known_station_ids: list, 
                known_station_names: list
                ) -> pd.DataFrame:    

                lats = self.data.columns.get_loc(f"{scenario}_lat")
                lngs = self.data.columns.get_loc(f"{scenario}_lng")

                rows_to_search = tqdm(
                    iterable=rows_missing_station_names_ids,
                    desc="Searching through rows to find matching latitudes and longitudes"
                )

                for row in rows_to_search:
                    for lat, lng in zip(known_lats, known_lngs):
                        if lat == self.data.iloc[row, lats] and lng == self.data.iloc[row, lngs]:
                            self.data = self.data.replace(
                                to_replace=self.data.iloc[row, station_id_col], 
                                value=known_station_ids[known_lats.index(lat)] 
                            )

                            self.data = self.data.replace(  
                                to_replace=self.data.iloc[row, station_name_col], 
                                value=known_station_names[known_lats.index(lat)] 
                            )
                return self.data

            self.data = __delete_rows_with_unnamed_and_missing_coordinates()
            
            if patient:       
                rows_missing_station_names_ids = __find_rows_with_missing_station_names_ids(scenario="end")
                known_lats, known_lngs, known_station_ids, known_station_names = _find_rows_with_known_coords_names_and_ids(scenario="end")
                
                self.data = _replace_missing_names_and_ids(
                        scenario="end",
                        known_lats=known_lats, 
                        known_lngs=known_lngs, 
                        known_station_ids=known_station_ids, 
                        known_station_names=known_station_names
                    )        

                self.data = self.data.drop(
                    columns=[
                        "ride_id", "rideable_type", "member_casual"
                    ]
                )

            else:
                self.data = self.data.drop(
                    columns=[
                        "ride_id", "rideable_type", "member_casual", "start_station_name", "end_station_name",
                        "start_station_id", "end_station_id" 
                    ]
                )

            if save:
                self.data.to_parquet(path=CLEANED_DATA/"cleaned.parquet") 

            return self.data

        else:
            logger.success("There is already some cleaned data. Fetching it...")
            return pd.read_parquet(path=CLEANED_DATA/"cleaned.parquet")

    def _make_training_data(self) -> None:
        """
        Extract raw data, transform it into a time series, and 
        transform that time series into training data which is 
        subsequently saved.
        """

        logger.info("Cleaning dataframe")
        self.data = self._clean()
        
        starts = self.data[
            ["start_time", "start_lat", "start_lng"]
        ]

        ends = self.data[
            ["end_time", "end_lat", "end_lng"]
        ]

        logger.info("Transforming the data into a time series...")
        agg_starts, agg_ends = self._transform_cleaned_data_into_ts_data(start_df=starts, end_df=ends)
        
        logger.info("Transforming time series data into training data...")
        trimmed_agg_data = {"start": agg_starts.iloc[:,:3], "end": agg_ends.iloc[:,:3]}
        
        for scenario in trimmed_agg_data.keys():
            training_data = self._transform_ts_into_training_data(
                ts_data=trimmed_agg_data[scenario],
                scenario=scenario,
                input_seq_len=24*28*1,
                step_size=24
            )

            logger.info("Saving the data so that we don't have to do this again...")
            training_data.to_parquet(path=TRAINING_DATA/f"{scenario}s.parquet")
    
    def _add_missing_slots(self, agg_data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Add rows to the input dataframe so that time slots with no
        trips are now populated in such a way that they now read as
        having 0 entries. This creates a complete time series.
        
        Args:
            agg_data (pd.DataFrame): the aggregate data
            scenario (str): "start"/"end"

        Returns:
            pd.DataFrame: t
        """
        self.station_ids = range(
            agg_data[f"{scenario}_station_id"].max() + 1
        )

        full_range = pd.date_range(
            start=agg_data[f"{scenario}_hour"].min(),
            end=agg_data[f"{scenario}_hour"].max(),
            freq="h"
        )

        output = pd.DataFrame()
        
        station_ids = tqdm(
            iterable=self.station_ids,
            desc=f"Categorising the data on the {scenario}s of trips by station IDs"
        )

        for station in station_ids:
            agg_data_i = agg_data.loc[
                agg_data[f"{scenario}_station_id"] == station, [f"{scenario}_hour", "trips"]
            ]

            # Create the indices for these new rows
            agg_data_i.index = pd.DatetimeIndex(agg_data_i.index)

            # Ensure that the number of indices is the same as the number of dates, filling empty timeslots with zeros
            agg_data_i = agg_data_i.reindex(full_range, fill_value=0)

            # Provide the station IDs
            agg_data_i[f"{scenario}_station_id"] = station
            output = pd.concat([output, agg_data_i])

        # Drop this column so that the next line doesn't result in two columns with the same name
        output = output.drop(f"{scenario}_hour", axis = 1) 

        # Rename the index appropriately
        output = output.reset_index().rename(
            columns={"index": f"{scenario}_hour"}
        )

        return output

    def _transform_cleaned_data_into_ts_data(
        self, 
        start_df: pd.DataFrame, 
        end_df: pd.DataFrame, 
        save: bool = True
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

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on the starts and ends  
                                               of trips.
        """

        def _get_ts_or_begin_transformation(starts_ts_path: str, ends_ts_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

            if Path(starts_ts_path).exists() and Path(ends_ts_path).exists():
                logger.success("Both time series datasets are already present")
                starts_ts = pd.read_parquet(path=starts_ts_path)
                ends_ts = pd.read_parquet(path=ends_ts_path)
                
            elif not Path(starts_ts_path).exists() and not Path(ends_ts_path).exists():
                logger.warning("Neither time series dataset exists")
                starts_ts, ends_ts = __begin_transformation(missing_scenario="both")

            elif not Path(starts_ts_path).exists() and Path(ends_ts_path).exists():
                logger.warning("Time series data for trip starts has not been made")
                ends_ts = pd.read_parquet(path=ends_ts_path)
                starts_ts = __begin_transformation(missing_scenario="start")
                
            elif Path(starts_ts_path).exists() and not Path(ends_ts_path).exists():
                logger.warning("Time series data for the ends of trips has not been made")
                starts_ts = pd.read_parquet(path=starts_ts_path)
                ends_ts = __begin_transformation(missing_scenario="end")
           
            return starts_ts, ends_ts

        def __begin_transformation(missing_scenario: str|None) -> tuple[pd.DataFrame, pd.DataFrame]|pd.DataFrame:

            dictionaries = []
            intermediate_dataframes = []
            ts_dataframes = []

            def ___round_coordinates_and_make_ids(data: pd.DataFrame, scenario: str, decimal_places: int) -> pd.DataFrame:
                logger.info(f"Recording the hour during which each trip {scenario}s...")

                data.insert(
                    loc=data.shape[1],
                    column=f"{scenario}_hour",
                    value=data.loc[:, f"{scenario}_time"].dt.floor("h"),
                    allow_duplicates=False
                )

                data = data.drop(f"{scenario}_time", axis=1)

                logger.info(f"Approximating the coordinates of the location where each trip {scenario}s...")
                # Round the latitudes and longitudes down to 5 dp, and add the rounded values as columns
                add_rounded_coordinates_to_dataframe(data=data, decimal_places=decimal_places, scenario=scenario)

                data = data.drop(
                    columns=[f"{scenario}_lat", f"{scenario}_lng"]
                )

                # Add the rounded coordinates to the dataframe as a column.
                add_column_of_rounded_points(data=data, scenario=scenario)

                data = data.drop(
                    columns=[f"rounded_{scenario}_lat", f"rounded_{scenario}_lng"]
                )

                intermediate_dataframes.append(data)
                logger.info("Matching up approximate locations with generated IDs...")

                # Make a list of dictionaries of start points and IDs
                origins_or_destinations_and_ids = make_dict_of_new_station_ids(data=data, scenario=scenario)
                dictionaries.append(origins_or_destinations_and_ids)

                # Critical for recovering the rounded coordinates and their corresponding IDs later.
                save_dict(
                    dictionary=origins_or_destinations_and_ids,
                    folder=GEOGRAPHICAL_DATA,
                    file_name=f"rounded_{scenario}_points_and_new_ids"
                )

                logger.success(f"Done with the {scenario}s of the trips!")
                return data

            def __aggregate_final_ts(data: pd.DataFrame, scenario: str) -> pd.DataFrame|list[pd.DataFrame, pd.DataFrame]:
                
                if missing_scenario == "start" or "end":
                    add_column_of_ids(data=data, scenario=scenario, points_and_ids=dictionaries[0])

                elif missing_scenario == "both":
                    if scenario == "start":
                        add_column_of_ids(data=data, scenario=scenario, points_and_ids=dictionaries[0])
                    elif scenario == "end":
                        add_column_of_ids(data=data, scenario=scenario, points_and_ids=dictionaries[1])

                data = data.drop(f"rounded_{scenario}_points", axis=1)

                logger.info(f"Aggregating the final time series data for the {scenario}s of trips...")
                agg_data = data.groupby([f"{scenario}_hour", f"{scenario}_station_id"]).size().reset_index()
                agg_data = agg_data.rename(columns={0: "trips"})

                if missing_scenario == "both":
                    ts_dataframes.append(self._add_missing_slots(agg_data=agg_data, scenario=scenario))
                    return ts_dataframes[0], ts_dataframes[1]
                    
                elif missing_scenario == "start" or "end":
                    return self._add_missing_slots(agg_data=agg_data, scenario=missing_scenario)
                    
            if missing_scenario == "both": 
                for data, scenario in zip(
                    [start_df, end_df], ["start", "end"]
                ):  
                    ___round_coordinates_and_make_ids(data=data, scenario=scenario, decimal_places=4)
                  
                    # Get all the coordinates that are common to both dictionaries
                    common_points = [point for point in dictionaries[0].keys() if point in dictionaries[1].keys()]

                    # Ensure that these common points have the same IDs in each dictionary.
                    for point in common_points:
                        dictionaries[0][point] = dictionaries[1][point]

                    for data, scenario in zip(
                            [intermediate_dataframes[0], intermediate_dataframes[1]], ["start", "end"]
                    ):
                        start_ts, end_ts = __aggregate_final_ts(data=data, scenario=scenario)

                        if save:
                            start_ts.to_parquet(TIME_SERIES_DATA/"starts_ts.parquet")
                            end_ts.to_parquet(TIME_SERIES_DATA/"ends_ts.parquet")
                
                return ts_dataframes[0], ts_dataframes[1]

            elif missing_scenario == "start" or "end":
                data = start_df if missing_scenario == "start" else end_df
                data = ___round_coordinates_and_make_ids(data=data, scenario=missing_scenario, decimal_places=4)
                ts_data = __aggregate_final_ts(data=data, scenario=missing_scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA/f"{missing_scenario}s_ts.parquet")

                return ts_data

        return _get_ts_or_begin_transformation(starts_ts_path=self.starts_ts_path, ends_ts_path=self.ends_ts_path)


    def _get_cutoff_indices(self, ts_data: pd.DataFrame, input_seq_len: int, step_size_len: int) -> list:
        """
        Starts by taking a certain number of rows of a given dataframe as an input, and the 
        indices of the row on which the selected rows starts and ends. These will be placed 
        in the first and second positions of a three element tuple. The third position of 
        said tuple will be occupied by the index of the row that comes after.

        Then the function will slide "step_size_len" steps and repeat the process. The function
        terminates once it reaches the last row of the dataframe.

        Credit to Pau Labarta Bajo

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

    def _transform_ts_into_training_data(
            self,
            ts_data: pd.DataFrame,
            scenario: str,
            input_seq_len: int,
            step_size: int
        ) -> tuple[pd.DataFrame, pd.Series]:

        """
        Transpose the time series data into a feature-target format.

        Args:

        Returns:
            tuple[pd.DataFrame, pd.Series]: 
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
            indices = self._get_cutoff_indices(
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
                x[i,:] = ts_data_per_station.iloc[index[0]: index[1]]["trips"].values
                y[i] = ts_data_per_station[index[1]:index[2]]["trips"].values[0]

                # Append the "hours" list with the appropriate entry at the intersection
                # of row "index[1]" and the hours column
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

            # Concatenate the dataframes
            features = pd.concat([features, features_per_location])
            targets = pd.concat([targets, targets_per_location])

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        
        training_data = pd.concat(
            [features, targets["trips_next_hour"]], axis=1
        )

        return training_data


if __name__ == "__main__":    
    make_fundamental_paths()    
    trips_2024 = DataProcessor(year=2024)
    trips_2024._make_training_data()
