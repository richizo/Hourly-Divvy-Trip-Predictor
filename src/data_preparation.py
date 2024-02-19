import pandas as pd 

from loguru import logger
from src.data_extraction import load_raw_data

from src.paths import TRAINING_DATA
from src.data_transformations import (
  clean_raw_data, transform_cleaned_data_into_ts_data, transform_ts_into_training_data
)


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
  
  jan_2024 = list(
    load_raw_data(year=2024, months=[1])
  )
  
  logger.info("Forming a dataframe")
  data = pd.concat(second_half_2023+jan_2024)
  
  logger.info("Cleaning said dataframe")
  clean_data = clean_raw_data(data)
  
  starts = clean_data[
    ["start_time", "start_latitude", "start_longitude"]
  ]
  
  stops = clean_data[
    ["stop_time", "stop_latitude", "stop_longitude"]
  ]
  
  logger.info("Transforming the data into a time series")
  agg_starts, agg_stops = transform_cleaned_data_into_ts_data(start_df = starts, stop_df = stops)
  
  
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