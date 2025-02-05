"""
Much of the code in this module is concerned with downloading the zip files that 
contain raw data, extracting their contents, and loading said contents as 
dataframes.

In an earlier version of this project, I downloaded the data for every year that 
Divvy has been in operation (from 2014 to date). 

I have since decided to restrict my training data to the most recent year of data so
because the most relevant data is obviously valuable, and because I don't want to deal 
with the extra demands on my memory and time that preprocessing that data would have required 
(not to speak of the training and testing of models).    
"""
import os
import requests
import pandas as pd

from pathlib import Path
from loguru import logger
from zipfile import ZipFile
from datetime import datetime as dt

from src.setup.paths import RAW_DATA_DIR, make_fundamental_paths

       
def download_and_extract_zipfile(year: int, month: int, keep_zipfile: bool = False) -> None:
    """
    Download the data for a given year, specifying the month if necessary,
    and the file name for the downloaded file.

    If the HTTP request for the data is successful, download the zipfile containing the data, 
    and extract the .csv file into a folder of the same name. The zipfile will be deleted by 
    default, unless otherwise specified.    

    Args:
        year (int): the year in question
        month (int, optional): the month for which we want that data. Defaults to None.
        keep_zipfile (bool, optional): whether the zipfile is to be kept after extraction. 
    """
    assert year >= 2021; 
    "The downloader is currently configured for years after 2021 because the zipfiles were packaged differently for earlier years."

    zipfile_name: str = f"{year}{month:02d}-divvy-tripdata.zip"
    url = f"https://divvy-tripdata.s3.amazonaws.com/{zipfile_name}"
    response = requests.get(url)

    if response.status_code == 200:
        file_name = zipfile_name[:-4]  # Remove ".zip" from the name of the zipfile
        folder_path = Path.joinpath(RAW_DATA_DIR, file_name)
        zipfile_path = Path.joinpath(RAW_DATA_DIR, zipfile_name)

        # Write the zipfile to the disk
        with open(file=zipfile_path, mode="wb") as zipfile:
            _ = zipfile.write(response.content)
        
        # Extract the contents of the zipfile
        with ZipFile(file=zipfile_path, mode="r") as zipfile:
            _ = zipfile.extract(f"{file_name}.csv", folder_path)  # Extract only the .csv file

        if not keep_zipfile:
            os.remove(zipfile_path)
    else:
        logger.error(f"Cannot find the zipfile. Status code: {response.status_code}")


def check_for_file_or_download(year: int, file_name: str, month: int | None = None):
    """
    Checks for the presence of a file, and downloads it if necessary.

    Args:
        year (int): the year whose data we are looking to potentially download
        file_name (str): the name of the file to be saved to disk
        month (list[int] | None, optional): the month for which we seek data
    """
    if month is not None:
        local_file = RAW_DATA_DIR/file_name
        if not local_file.exists():
            try:
                logger.info(f"Downloading and extracting {file_name}.zip")
                download_and_extract_zipfile(year=year, month=month)
            except Exception as error:
                logger.error(error)
        else:
            logger.success(f"{file_name}.zip is already saved")


def get_dataframe_from_folder(file_name: str) -> pd.DataFrame:
    """
    Load a requested data file which has been downloaded and return it as dataframes.

    Args:
        file_name (str): the name of the file to be loaded.

    Returns:
        pd.DataFrame: the loaded dataframe 
    """
    data = pd.DataFrame()
    data_one_month = pd.read_csv(RAW_DATA_DIR/f"{file_name}/{file_name}.csv")
    data = pd.concat([data, data_one_month], axis=0)
    return data


def load_raw_data(year: int, months: list[int] | None = None) -> pd.DataFrame:
    """
    Download or load the data for either the specified months of the year in question, or 
    for all months up to the present month (if the data being sought is from this year).
    
    Args:
        year (int): the year whose data we want to load
        months (list[int] | None, optional): the months for which we want data

    Yields:
        Iterator[pd.DataFrame]: the requested datasets.
    """
    make_fundamental_paths()
    is_current_year = True if year == dt.now().year else False
    end_month = dt.now().month if is_current_year else 12
    months_to_download = range(1, end_month + 1) if months is None else months

    for month in months_to_download:
        file_name = f"{year}{month:02d}-divvy-tripdata"

        try:
            check_for_file_or_download(year=year, month=month, file_name=file_name)
            yield get_dataframe_from_folder(file_name=file_name)
        except Exception as error:
            logger.error(error)
           
