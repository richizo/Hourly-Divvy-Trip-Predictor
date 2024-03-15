import numpy as np
import pandas as pd

from typing import Optional
from tqdm import tqdm

from loguru import logger

from src.setup.paths import TRAINING_DATA

from src.setup.miscellaneous import (
    add_column_of_rounded_points, 
    add_column_of_ids, make_new_station_ids, 
    add_rounded_coordinates_to_dataframe, save_dict
)


def clean_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This is the same cleaning process used in the notebook named "03_DATA_CLEARNING.ipynb"
    It works for 2023 data, but may not necessarily be ideal for later years if the datasets
    for those years is structured differently.
    """

    data["started_at"] = pd.to_datetime(data["started_at"], format="mixed")
    data["ended_at"] = pd.to_datetime(data["ended_at"], format="mixed")

    data = data.drop(
        columns=[
            "ride_id", "rideable_type", "member_casual", "start_station_id", "end_station_id",
            "start_station_name", "end_station_name"
        ]
    )

    data = data.rename(
        columns={
            "started_at": "start_time",
            "ended_at": "stop_time",
            "start_lat": "start_latitude",
            "start_lng": "start_longitude",
            "end_lat": "stop_latitude",
            "end_lng": "stop_longitude",
        }
    )

    data = data.dropna()
    data.drop_duplicates(inplace=True)

    return data


def add_missing_slots(
    agg_data: pd.DataFrame,
    scenario: str 
) -> pd.DataFrame:
    """
    Add rows to the input dataframe so that time slots with no
    trips are now populated in such a way that they now read as
    having 0 entries. This creates a complete time series.
    """

    station_ids = range(
        agg_data[f"{scenario}_station_id"].max() + 1
    )

    full_range = pd.date_range(
        agg_data[f"{scenario}_hour"].min(),
        agg_data[f"{scenario}_hour"].max(),
        freq="h"
    )

    output = pd.DataFrame()

    for station in tqdm(station_ids):

        agg_data_i = agg_data.loc[
            agg_data[f"{scenario}_station_id"] == station, [f"{scenario}_hour", "trips"]
        ]

        agg_data_i.set_index(f"{scenario}_hour", inplace=True)

        if agg_data_i.empty:

            # Add a missing dates with zero rides
            agg_data_i = pd.DataFrame(
                data=[{
                    f"{scenario}_hour": agg_data[f"{scenario}_hour"].max(), "trips": 0
                }]
            )

        # Create the indices for these new rows
        agg_data_i.index = pd.DatetimeIndex(agg_data_i.index)

        # Ensure that the number of indices is the same as the number of dates in the full range above, 
        # and provide the value of zero (for trips) for all times at which there are no trips.

        agg_data_i = agg_data_i.reindex(full_range, fill_value=0)

        agg_data_i[f"{scenario}_station_id"] = station

        output = pd.concat([output, agg_data_i])

    output = output.reset_index().rename(columns={"index": f"{scenario}_hour"})

    return output


def transform_cleaned_data_into_ts_data(
    start_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    year: Optional[int] = None
):
    """
    This function contains all the code in the homonymous notebook, however it has some
    distinguishing features to enable it to integrate with other functions in the pipeline.

    For one thing, it accepts two dataframes. Those dataframes should be:
    - one that consists of the "start_time", "start_latitude", and "start_longitude" columns.
    - another that consists of the "stop_time", "stop_latitude", and "stop_longitude" columns.
    """

    from src.paths import GEOGRAPHICAL_DATA
    from loguru import logger
    
    dictionaries = []
    intermediate_dataframes = []
    final_dataframes = []
    
    logger.info("This will take a while")

    for data, scenario in zip(
            [start_df, stop_df], ["start", "stop"]
    ):

        logger.info(f"Computing the hours during which each trip {scenario}s")

        data.insert(
            loc=data.shape[1],
            column=f"{scenario}_hour",
            value=data.loc[:, f"{scenario}_time"].dt.floor("h"),
            allow_duplicates=False
        )

        data = data.drop(
            columns=[
                f"{scenario}_time", f"{scenario}_time"
            ]
        )

        logger.info(f"Approximating the coordinates of the location at which each trip {scenario}s")
        
        # Round the latitudes and longitudes down to 4 dp, and add the rounded values as columns
        add_rounded_coordinates_to_dataframe(data=data, decimal_places=4, scenario=scenario)

        data = data.drop(
            columns=[
                f"{scenario}_latitude", f"{scenario}_longitude"
            ]
        )

        # Add the rounded coordinates to the dataframe as a column.
        add_column_of_rounded_points(data=data, scenario=scenario)

        data = data.drop(
            columns=[
                f"rounded_{scenario}_latitude", f"rounded_{scenario}_longitude"
            ]
        )

        intermediate_dataframes.append(data)

        logger.info("Matching up approximate locations with generated IDs")
        
        if scenario == "start": 
            
            if year == 2024:
                
                import os
                import pickle

                with open(GEOGRAPHICAL_DATA/"rounded_origin_points_and_new_ids", "rb") as f:
                
                    origins_and_ids = pickle.load(f)
                    
                dictionaries.append(origins_and_ids)
                
            else:
            
                # Make a list of dictionaries of start points and IDs
                origins_and_ids = make_new_station_ids(data=data, scenario=scenario)
                dictionaries.append(origins_and_ids)
                
                # This (and its counterpart below) is critical for recovering the rounded coordinates 
                # later on, and for knowing which IDs they correspond to.
                save_dict(
                    dictionary=origins_and_ids,
                    folder=GEOGRAPHICAL_DATA,
                    file_name="rounded_origin_points_and_new_ids"
                )
                
                
        elif scenario == "stop":
            
            if year == 2024:
                
                import os
                import pickle

                with open(GEOGRAPHICAL_DATA/"rounded_destination_points_and_new_ids", "rb") as f:
                
                    destinations_and_ids = pickle.load(f)
                
            else:
                    
                # Make a list of dictionaries of stop points and IDs
                destinations_and_ids = make_new_station_ids(data=data, scenario=scenario)
                dictionaries.append(destinations_and_ids)
                
                save_dict(
                    dictionary=destinations_and_ids,
                    folder=GEOGRAPHICAL_DATA,
                    file_name="rounded_destination_points_and_new_ids"
                )

        print("\n")

    # Get all the coordinates that are common to both dictionaries
    common_points = [
        point for point in dictionaries[0].keys() if point in dictionaries[1].keys()
    ]

    # Ensure that these common points have the same IDs in each dictionary.
    for point in common_points:

        dictionaries[0][point] = dictionaries[1][point]

    for data, scenario in zip(
            [intermediate_dataframes[0], intermediate_dataframes[1]], ["start", "stop"]
    ):

        if scenario == "start":
            add_column_of_ids(data=data, scenario=scenario, points_and_ids=dictionaries[0])

        if scenario == "stop":
            add_column_of_ids(data=data, scenario=scenario, points_and_ids=dictionaries[1])

        data = data.drop(f"rounded_{scenario}_points", axis=1)

        logger.info(f"Aggregating the final data on trip {scenario}s")

        agg_data = data.groupby([f"{scenario}_hour", f"{scenario}_station_id"]).size().reset_index()
        agg_data = agg_data.rename(columns={0: "trips"})

        final_dataframes.append(
            add_missing_slots(agg_data=agg_data, scenario=scenario)
        )
        
    return final_dataframes[0], final_dataframes[1]


def get_cutoff_indices(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size_len: int
):
    """This function will take a certain number of rows of a given dataframe as an input,
    and take the indices of the row on which it starts and ends. These will be placed in the
    first and second positions of a three element tuple. The third position of said tuple
    will be occupied by the index of the row that comes after.

    Then the function will slide "step_size_len" steps and repeat the process. The function
    terminates once it reaches the last row of the dataframe.

    Credit to Pau Labarta Bajo
    """

    # The function has to stop at the last row of the dataframe
    stop_position = len(ts_data) - 1

    # These numbers will be the first, second, and third elements of each tuple of indices.
    subseq_first_index = 0
    subseq_mid_index = input_seq_len
    subseq_last_index = input_seq_len + 1

    # This will be the list of indices
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
    ts_data: pd.DataFrame,
    scenario: str,
    input_seq_len: int,
    step_size: int
) -> tuple[pd.DataFrame, pd.Series]:

    """ Transpose the time series data into a feature-target format."""

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
        ].sort_values(
            by=[f"{scenario}_hour"]
        )

        # Compute cutoff indices
        indices = get_cutoff_indices(
            ts_data=ts_data_per_station,
            input_seq_len=input_seq_len,
            step_size_len=step_size
        )

        n_examples = len(indices)

        # Create a multidimensional array for the features, and a column vector for the target
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=n_examples, dtype=np.float32)

        hours = []
        for i, index in enumerate(indices):
            x[i, :] = ts_data_per_station.iloc[index[0]: index[1]]["trips"].values
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

    return features, targets["trips_next_hour"]


def make_training_data(scenario: str) -> pd.DataFrame:
  """
  Extract raw data, transforma it into a time series, and 
  transform that time series into training data.

  Returns:
      pd.DataFrame: the training dataset for start or stop data
                    (as the case may be)
  """
  
  logger.info("Fetching raw data from the Divvy site")
  
  second_half_2023 = list(
    load_raw_data(year=2023, months = list(range(6,13)))
  )
  
  jan_feb_2024 = list(
    load_raw_data(year=2024, months=[1,2])
  )
  
  logger.info("Forming a dataframe")
  data = pd.concat(second_half_2023+jan_feb_2024)
  
  logger.info("Cleaning said dataframe")
  clean_data = clean_raw_data(data)
  
  starts = clean_data[
    ["start_time", "start_latitude", "start_longitude"]
  ]
  
  stops = clean_data[
    ["stop_time", "stop_latitude", "stop_longitude"]
  ]
  
  logger.info("Transforming the data into a time series")
  agg_starts, agg_stops = transform_cleaned_data_into_ts_data(
    start_df = starts, 
    stop_df = stops
  )
  
  logger.info("Transforming time series into training data")
  
  if scenario == "start":
    
    trimmed_agg_data = agg_starts.iloc[:,:3]
    
    start_features, start_target = transform_ts_into_training_data(
      ts_data=trimmed_agg_starts,
      start_or_stop="start",
      input_seq_len=24*28*1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
      step_size=24
    )                                                   
  
    start_features["trips_next_hour"] = start_target
    start_table = start_features
    
    logger.info("Saving the data so we don't have to do this again")
    start_table.to_parquet(path=TRAINING_DATA/"starts.parquet")

    return start_table
  
  
  elif scenario == "stop":
    
    trimmed_agg_stops = agg_stops.iloc[:,:3]      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    stop_features, stop_target = transform_ts_into_training_data(
      ts_data=trimmed_agg_stops,
      start_or_stop="stop",
      input_seq_len=24*28*1, 
      step_size=24
    )
  
    stop_features["trips_next_hour"] = stop_target  
    stop_table = stop_features      
    
    logger.info("Saving the data so we don't have to do this again")
    stop_table.to_parquet(path=TRAINING_DATA/"stops_parquet")
    
    return stop_table 
    