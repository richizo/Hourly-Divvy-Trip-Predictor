import sys
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim, Photon

sys.path.insert(0, "/home/kobina/cyclistic-bike-sharing-data/src")


# SAVING DATA
def concat_to_parquet(
        list_of_dataframes: list,
        parquet_name: str,
        folder_name: str
):
    """
    This function takes the elements of a list of dataframes, 
    concatenates them, and returns the result as a .parquet file.
    """

    return pd.concat(list_of_dataframes).to_parquet(
        path=f"{folder_name}/{parquet_name}.parquet"
    )


def save_as_parquet(
        list_of_dataframes: list,
        folder_name: str):
    """
    This function takes a list of dataframes and saves each as
    a .pkl file, placing it in a specified folder. Each .pkl 
    file is named according to its position in the list of
    dataframes that it came from.
    """

    for i in tqdm(enumerate(list_of_dataframes)):
        i[1].to_parquet(path=f"{folder_name}/{i[0]}.parquet")


# MEMORY MANAGEMENT

def view_memory_usage(
        data: pd.DataFrame,
        column: str):
    """
    This function allows us to view the amount of memory being 
    used by one or more columns of a given dataframe.
    """

    yield data[column].memory_usage(index=False, deep=True)


def change_column_data_type(
        data: pd.DataFrame,
        columns: list,
        to_format: str):
    """
    This function changes the datatype of one or more columns of 
    a given dataframe.
    """

    data[columns] = data[columns].astype(to_format)


# DATA CLEANING

def find_first_nan(
        data: pd.DataFrame,
        missing: bool,
        just_reveal: bool
):
    """
    When "missing" is set to True, this function will look
    through the first column of the dataframe, and tell us
    on which row a missing value first occurs.

    When "missing" is set to False, the function tells 
    us on which row a non-missing value first occurs.
    """

    for i in tqdm(range(len(data))):

        if pd.isnull(data.iloc[i, 0]) == missing:

            if just_reveal is True:

                print(i)
                break

            else:
                return i
                break


# GEOCODING

def use_primary_geocoder(places: list,):
    """
    This function initialises the Nominatim geocoder, and takes a list 
    of place names. It then generates a precise location using the geocoder, 
    and creates key, value pairs of each place name with its respective 
    location. 

    Some of the places will not be successfully processed by the geocoder, 
    causing it (the geocoder) to return "None". For each of these, I 
    would like the function to provide the string below as the value
    corresponding to the place name (the key). The reason why a 
    string was used instead of the default "None" is that "None" as 
    a data type is non-iterable.
    """

    places_and_points = {}
    geolocator = Nominatim(user_agent="maadabrandon@protonmail.com")

    for place in tqdm(places):

        if geolocator.geocode(place, timeout=120) is None:

            places_and_points[f"{place}"] = (0,0)

        else:

            places_and_points[f"{place}"] = geolocator.geocode(place, timeout=None)[-1]

    return places_and_points


def use_secondary_geocoder(
        data: pd.DataFrame,
        column_of_station_names: str,
        row_indices: list,

):

    non_geocoded_places = list(
        data.iloc[
            row_indices, data.columns.get_loc(column_of_station_names)
        ].unique()
    )

    remaining_places_with_points = {}
    geolocator = Photon()

    for place in tqdm(non_geocoded_places):

        if geolocator.geocode(place) is None:
            remaining_places_with_points[f"{place}"] = (0, 0)

        else:
            remaining_places_with_points[f"{place}"] = geolocator.geocode(place, timeout=1000)[-1]

    return remaining_places_with_points


def add_coordinates_to_dataframe(
        data: pd.DataFrame,
        places_and_points: dict,
        start_or_stop: str
):

    """
    After forming the dictionary of places and coordinates, this function isolates
    the latitudes, and longitudes and place them in named appropriately named
    columns of a target dataframe.
    """

    if start_or_stop == "start":
        points = [
            places_and_points[place] for place in data["from_station_name"] if place in places_and_points.keys()
        ]

    if start_or_stop == "stop":
        points = [
            places_and_points[place] for place in data["to_station_name"] if place in places_and_points.keys()
        ]

    data[f"{start_or_stop}_latitude"] = pd.Series([point[0] for point in points])
    data[f"{start_or_stop}_longitude"] = pd.Series([point[1] for point in points])


# FIND COLUMNS THAT CONTAIN STRINGS
def find_rows_with_zeros(
        data: pd.DataFrame,
        column_index: int
):

    return [
        row for row in tqdm(range(data.shape[0])) if data.iloc[row, column_index] == 0
    ]


def reveal_final_unknown_lats(
        data: pd.DataFrame,
        column_of_coordinate: str
):

    return [
            row for row in tqdm(range(data.shape[0])) if data.iloc[
                                                             row, data.columns.get_loc(column_of_coordinate)] == 0
    ]
