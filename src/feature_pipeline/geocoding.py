import pandas as pd 
from geopy.geocoders import Nominatim, Photon


def use_primary_geocoder(places: list,):
    """
    This function initialises the Nominatim geocoder, and takes a list 
    of place names. It then generates a precise location using the geocoder, 
    and creates key, value pairs of each place name with its respective 
    location. 

    Some of the locations will not be successfully processed by the geocoder, 
    causing it (the geocoder) to return "None". For each of these, I 
    would like the function to provide the string below as the value
    corresponding to the place name (the key). The reason why a 
    string was used instead of the default "None" is that "None" as 
    a data type is non-iterable.
    """

    places_and_points = {}
    geolocator = Nominatim(user_agent=settings.email)

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
        scenario: str
):

    """
    After forming the dictionary of places and coordinates, this function isolates
    the latitudes, and longitudes and place them in named appropriately named
    columns of a target dataframe.
    """

    if scenario == "start":
        points = [
            places_and_points[place] for place in data["from_station_name"] if place in places_and_points.keys()
        ]

    if scenario == "stop":
        points = [
            places_and_points[place] for place in data["to_station_name"] if place in places_and_points.keys()
        ]

    data[f"{scenario}_latitude"] = pd.Series(
        [point[0] for point in points]
    )
    
    data[f"{scenario}_longitude"] = pd.Series(
        [point[1] for point in points]
    )
