"""
Much of the code in this module is concerned with downloading the zip files that 
contain raw data, extracting their contents, and loading said contents as 
dataframes.

In an earlier version of this project, I downloaded the data for every year that 
Divvy has been in operation, resulting in the code below. I have since decided to
restrict my training data to data from 2023 to mid-2024. I did this because it would
take a massive amount of memory and time to handle the creation of features and 
targets, let along the testing and training of models.    

I could delete the parts of this function that pertain to all the previous years, 
but I haven't had the heart to, because getting this working for all past years was 
quite difficult.
"""
import os
import requests
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from loguru import logger
from zipfile import ZipFile
from datetime import datetime as dt

from src.setup.paths import RAW_DATA_DIR


def download_one_file_of_raw_data(
    year: int,
    month: int = None,
    quarters: list[int] = None,
    file_name: str = None
) -> None:
    """
    Download the data for a given year, specifying the month and quarters if necessary,
    and the file name for the downloaded file.
    """

    def __download_and_extract_zipfile(
        first_zipfile_name: str,
        URL_1: str, 
        second_zipfile_name: str = None, 
        URL_2: str = None
    ) -> None:  

        def __write_and_extract_zipfile(
            zipfile_name: str, 
            response: requests.Response, 
            keep_zipfile: bool = False
        ) -> None:
            file_name = zipfile_name[:-4] # Remove ".zip" from the name of the zipfile
            folder_path = RAW_DATA_DIR/file_name
            zipfile_path = RAW_DATA_DIR/zipfile_name
            open(file=zipfile_path, mode="wb").write(response.content)

            with ZipFile(file=zipfile_path, mode="r") as zip:
                zip.extract(f"{file_name}.csv", folder_path) # Extract only the the .csv file
            if not keep_zipfile:    
                os.remove(zipfile_path)

        first_response = requests.get(URL_1)    
        __write_and_extract_zipfile(zipfile_name=first_zipfile_name, response=first_response)

        if URL_2 and second_zipfile_name is not None:
            second_response = requests.get(URL_2)
            __write_and_extract_zipfile(zipfile_name=second_zipfile_name, response=second_response)


    if year == 2014:
        URL_1 = f"https://divvy-tripdata.s3.amazonaws.com/Divvy_Stations_Trips_{year}_Q{quarters[0]}Q{quarters[1]}.zip"
        URL_2 = f"https://divvy-tripdata.s3.amazonaws.com/Divvy_Stations_Trips_{year}_Q{quarters[2]}Q{quarters[3]}.zip"

        __download_and_extract_zipfile(
            zipfile_name=f"Divvy_Trips_{year}_Q1Q2.zip", 
            URL=URL_1,
            second_zipfile_name=f"Divvy_Stations_Trips_{year}_Q3Q4.zip",
            URL_2=URL_2
        )
            
    if year in [2015, 2016, 2017]:
        
        if file_name in [f"Divvy_Trips_{year}-Q1Q2", f"Divvy_Trips_{year}_Q3Q4"]:
            URL = f"https://divvy-tripdata.s3.amazonaws.com/{file_name}.zip"
            __download_and_extract_zipfile(first_zipfile_name=f"{file_name}.zip", URL_1=URL)

    if year in [2018, 2019]:

        for quarter in quarters:
            URL = f"https://divvy-tripdata.s3.amazonaws.com/Divvy_Trips_{year}_Q{quarter}.zip"
            __download_and_extract_zipfile(first_zipfile_name=f"Divvy_Trips_{year}_Q{quarter}.zip", URL_1=URL)

    if year == 2020:

        if quartjers == [1]:
            URL = f"https://divvy-tripdata.s3.amazonaws.com/Divvy_Trips_{year}_Q1.zip"
            __download_and_extract_zipfile(first_zipfile_name=f"Divvy_Trips_{year}_Q1.zip", URL_1=URL)

        if month >= 4:
            URL = f"https://divvy-tripdata.s3.amazonaws.com/{year}{month:02d}-divvy-tripdata.zip"
            __download_and_extract_zipfile(first_zipfile_name=f"{year}{month:02d}-divvy-tripdata.zip", URL_1=URL)

    if year >= 2021:
        URL = f"https://divvy-tripdata.s3.amazonaws.com/{year}{month:02d}-divvy-tripdata.zip"
        __download_and_extract_zipfile(first_zipfile_name=f"{year}{month:02d}-divvy-tripdata.zip", URL_1=URL)


def check_for_file_and_download(
    year: int,
    file_name: str,
    month: int = None,
    quarters: list[int] = None
):
    """
    Checks for the presence of a file, and downloads if necessary.

    Args:
        year (int): the year whose data we are looking to potentially download

        file_name (str): the name of the file to be saved to disk
        
        months (list[int] | None, optional): the month for which we seek data

        quarters (list[int] | None, optional): the quarters of the year (when 
                                               the data is saved in quarterly 
                                               chunks)
    """
    if quarters is not None:
        local_file = RAW_DATA_DIR/file_name
        if not local_file.exists():
            try:
                logger.info(f"Downloading and extracting {file_name}.zip")
                download_one_file_of_raw_data(year=year, file_name=file_name, quarters=quarters)
            except:
                logger.error(f"{file_name} is not available")
        else:
            logger.success(f"{file_name} is already in local storage")

    elif month is not None:
        local_file = RAW_DATA_DIR/f"{year}{month:02d}-divvy-tripdata"
        if not local_file.exists():
            try:
                logger.info(f"Downloading and extracting {year}{month:02d}-divvy-tripdata.zip")
                download_one_file_of_raw_data(year=year, month=month)
            except:
                logger.error(f"{year}{month:02d}-divvy-tripdata is not available")
        else:
            logger.success(f"The file {year}{month:02d}-divvy-tripdata.zip is already in local storage")


def get_dataframe_from_folder(year: int, file_name: str) -> pd.DataFrame:
    """
    Load a requested data file which has been downloaded and return it as dataframes.

    Args:
        year (int): the year whose data is to be loaded.

        file_name (str): the name of the file to be loaded.

    Returns:
        pd.DataFrame: the loaded dataframe 
    """
    data = pd.DataFrame()

    if year == 2014:
        if file_name == f"Divvy_Trips_{year}_Q1Q2":
            data_q1_q2 = pd.read_csv(RAW_DATA_DIR/f"{file_name}/{file_name}.csv")
            data = pd.concat([data, data_q1_q2])

        elif file_name == f"Divvy_Trips_{year}_Q3Q4":
            other_months = pd.read_csv(RAW_DATA_DIR/f"Divvy_Stations_Trips_2014_Q3Q4/{file_name}.csv")
            data = pd.concat([data, other_months])

    elif year == 2015:
        if file_name == f"Divvy_Trips_{year}-Q1Q2":
            for quarter in [1, 2]:
                quarter_data = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}-Q{quarter}.csv")
                data = pd.concat([data, quarter_data])

        elif file_name == f"Divvy_Trips_{year}_Q3Q4":
            for month in [7, 8, 9]:
                intermediate_month = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}_{month:02d}.csv")
                data = pd.concat([data, intermediate_month])

            last_quarter = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}_Q4.csv")
            data = pd.concat([data, last_quarter])

    elif year == 2016:
        if file_name == f"Divvy_Trips_{year}_Q1Q2":
            first_quarter = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}_Q1.csv")
            for month in [4, 5, 6]:
                second_quarter_month = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}_{month:02d}.csv")
                data = pd.concat([data, first_quarter, second_quarter_month])

        elif file_name == f"Divvy_Trips_{year}_Q3Q4":
            for quarter in [3, 4]:
                quarter_data = pd.read_csv(RAW_DATA_DIR / f"{file_name}/Divvy_Trips_{year}_Q{quarter}.csv")
                data = pd.concat([data, quarter_data])

    elif year == 2017:
        if file_name == f"Divvy_Trips_{year}_Q1Q2":
            for quarter in [1, 2]:
                quarter_data = pd.read_csv(RAW_DATA_DIR/f"{file_name}/Divvy_Trips_{year}_Q{quarter}.csv")
                data = pd.concat([data, quarter_data])

        if file_name == f"Divvy_Trips_{year}_Q3Q4":
            for quarter in [3, 4]:
                quarter_data = pd.read_csv(RAW_DATA_DIR /f"{file_name}/Divvy_Trips_{year}_Q{quarter}.csv")
                data = pd.concat([data, quarter_data])

    else:
        data_one_month = pd.read_csv(RAW_DATA_DIR/f"{file_name}/{file_name}.csv")
        data = pd.concat([data, data_one_month])

    return data


def load_raw_data(year: int, months: list[int] = None, quarters: list[int] = None) -> pd.DataFrame:
    """
    Check for the presence of the specified data file, download it if it is absent, and load the 
    data from disk.

    Args:
        year (int): the year whose data we want to load
        months (list[int] | None, optional): applicable for years when the data has been split by month.
        quarters (list[int] | None, optional): applicable for years when the data has been split by quarters.

    Yields:
        Iterator[pd.DataFrame]: the requested dataset.
    """

    if year < 2014:
        raise Exception("There is no source data before 2014.")

    if year == 2014:
        file_names_2014 = [f"Divvy_Trips_{year}_Q1Q2", f"Divvy_Trips_{year}-Q3Q4"]

        for file_name in file_names_2014:
            check_for_file_and_download(year=year, file_name=file_name, quarters=quarters)
            yield get_dataframe_from_folder(year=year, file_name=file_name)

    elif year == 2015:
        for file_name in [f"Divvy_Trips_{year}-Q1Q2", f"Divvy_Trips_{year}_Q3Q4"]:
            check_for_file_and_download(year=year, file_name=file_name, quarters=[1])
            yield get_dataframe_from_folder(year=year, file_name=file_name)

    elif year == 2016:
        check_for_file_and_download(year=year, file_name=f"Divvy_Trips_{year}_Q1Q2", quarters=[1, 2])
        check_for_file_and_download(year=year, file_name=f"Divvy_Trips_{year}_Q3Q4", quarters=[3, 4])

        for file_name in [f"Divvy_Trips_{year}_Q1Q2", f"Divvy_Trips_{year}_Q3Q4"]:
            yield get_dataframe_from_folder(year=year, file_name=file_name)

    elif year == 2017:
        check_for_file_and_download(year=year, file_name=f"Divvy_Trips_{year}_Q1Q2", quarters=[1, 2])
        check_for_file_and_download(year=year, file_name=f"Divvy_Trips_{year}_Q3Q4", quarters=[3, 4])

        for file_name in [f"Divvy_Trips_{year}_Q1Q2", f"Divvy_Trips_{year}_Q3Q4"]:
            yield get_dataframe_from_folder(year=year, file_name=file_name)

    elif year in [2018, 2019]:
        for quarter in quarters:
            check_for_file_and_download(year=year, file_name=f"Divvy_Trips_{year}_Q{quarter}", quarters=[quarter])
            yield get_dataframe_from_folder(year=year, file_name=f"Divvy_Trips_{year}_Q{quarter}")

    elif year == 2020 and quarters == [1]:
        check_for_file_and_download(year=year, quarters=quarters, file_name=f"Divvy_Trips_{year}_Q1")
        yield get_dataframe_from_folder(year=year, file_name=f"Divvy_Trips_{year}_Q1")

    elif year == 2020 and quarters is None:
        months = range(4, 13)
        for month in tqdm(months):
            check_for_file_and_download(year=year, month=month, file_name=f"{year}{month:02d}-divvy-tripdata")
            yield get_dataframe_from_folder(year=year, file_name=f"{year}{month:02d}-divvy-tripdata")

    elif year >= 2021 and year != dt.utcnow().year:
        if months is None:
            for month in range(1, 13):
                check_for_file_and_download(year=year, month=month, file_name=f"{year}{month:02d}-divvy-tripdata")
                yield get_dataframe_from_folder(year=year, file_name=f"{year}{month:02d}-divvy-tripdata")

        else:
            for month in months:
                check_for_file_and_download(year=year, month=month, file_name=f"{year}{month:02d}-divvy-tripdata")
                yield get_dataframe_from_folder(year=year, file_name=f"{year}{month:02d}-divvy-tripdata")
    
    elif year == dt.now().year:
        if months is None:
            for month in range(1, dt.now().month+1):
                
                try: 
                    check_for_file_and_download(year=year, month=month, file_name=f"{year}{month:02d}-divvy-tripdata")
                    yield get_dataframe_from_folder(year=year, file_name=f"{year}{month:02d}-divvy-tripdata")
                except:
                    break   
