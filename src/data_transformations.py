import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Optional, List, Tuple


def add_missing_slots(
        agg_data: pd.DataFrame,
        start_or_stop: str
) -> pd.DataFrame:
    """
    Add rows to the input dataframe so that time slots with no
    trips are now populated in such a way that they now read as
    having 0 slots. This creates a complete time series.
    """

    station_ids = agg_data[f"{start_or_stop}_station_id"].unique()

    full_range = pd.date_range(
        agg_data[f"{start_or_stop}_hour"].min(),
        agg_data[f"{start_or_stop}_hour"].max(),
        freq="H"
    )

    output = pd.DataFrame()

    for station in tqdm(station_ids):

        agg_data_i = agg_data[agg_data[f"{start_or_stop}_station_id"] == station]

        if agg_data_i.empty:
            # In an hour with no trips, make a row where there are zero trips in the given hour.
            agg_data_i = pd.DataFrame.from_dict(
                {
                    f"{start_or_stop}_hour": agg_data[f"{start_or_stop}_hour"].max(), "trips": 0
                }
            )

        # Set the index
        agg_data_i.set_index(f"{start_or_stop}_hour", inplace=True)

        # Create the indices for these new rows
        agg_data_i.index = pd.DatetimeIndex(agg_data_i.index)

        # Ensure that the number of indices is the same as the number of dates in the full range above, 
        # and provide the value of zero (for trips) for all times at which there are no trips.

        agg_data_i = agg_data_i.reindex(full_range, fill_value=0)

        agg_data_i[f"{start_or_stop}_station_id"] = station

        output = pd.concat([output, agg_data_i])

    output = output.reset_index().rename(columns={"index": f"{start_or_stop}_hour"})

    return output


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
        start_or_stop: str,
        input_seq_len: int,
        step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """" Transpose the time series data into a feature-target format."""

    # Ensure first that these are the columns of the chosen data set (and they are listed in this order)
    assert set(ts_data.columns) == {f"{start_or_stop}_hour", f"{start_or_stop}_station_id", "trips"}

    station_ids = ts_data[f"{start_or_stop}_station_id"].unique()

    # Prepare the dataframe which will contain the features and targets
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for station_id in tqdm(station_ids):

        # Isolate a part of the dataframe that relates to each station ID
        ts_data_per_station = ts_data.loc[
            ts_data[f"{start_or_stop}_station_id"] == station_id, [f"{start_or_stop}_hour", "trips"]
        ].sort_values(
            by=[f"{start_or_stop}_hour"]
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
                ts_data_per_station.iloc[index[1]][f"{start_or_stop}_hour"]
            )

        # Make a dataframe of features
        features_per_location = pd.DataFrame(
            x, columns=[
                f"rides_previous_{i + 1}_hour" for i in reversed(range(input_seq_len))
            ]
        )

        features_per_location[f"{start_or_stop}_hour"] = hours
        features_per_location[f"{start_or_stop}_station_id"] = station_id

        targets_per_location = pd.DataFrame(y, columns=["trips_next_hour"])

        # Concatenate the dataframes
        features = pd.concat([features, features_per_location])
        targets = pd.concat([targets, targets_per_location])

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)

        return features, targets["trips_next_hour"]
