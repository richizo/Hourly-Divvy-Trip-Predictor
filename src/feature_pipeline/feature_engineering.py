import pandas as pd

from loguru import logger
from warnings import simplefilter
from geopy.geocoders import Nominatim, Photon

from src.setup.config import config


class GeoData:
    """
    The code that makes up what is now this class was created in order to geocode an older version
    of this dataset which did not contain the coordinates of each station. In 2024's data, coordinates are
    provided, so we will not be geocoding. However, if Divvy fails to provide coordinates in future datasets,
    this code may be of use.
    """
    def __init__(self, data: pd.DataFrame, scenario: str) -> None:
        self.data = data
        self.scenario = scenario
        self.latitudes = []
        self.longitudes = []
        self.place_names = self.data[f"{scenario}_station_name"].unique().to_list()

    def geocode(self) -> dict:
        """
        Initialises the Nominatim geocoder, and applies it to a list of place names.
        It then generates the coordinates of each place and creates key, value pairs
        of each place name with its respective coordinate.

        Some of the locations will not be successfully processed by the geocoder,
        causing it (the geocoder) to return None. For each such case, the function
        provides (0,0) as the value corresponding to the place name. After this,
        the locations that were unable to be geocoded by Nominatim will be run
        through the Photon geocoder. This will (hopefully) result in the geocoding
        many of these locations. Those that are unsuccessfully geocoded will again
        have (0,0) as their corresponding coordinates.


        Returns:
            dict: a dictionary which contains key value pairs of place names and
                  coordinates
        """

        def _trigger_geocoder(geocoder: Nominatim | Photon, place_names: list) -> dict:
            """

            Args:
                geocoder: the geocoder to be used.
                place_names: the names of the places that are to be geocoded.

            Returns:

            """
            places_and_points = {}
            for place in place_names:
                if place in places_and_points.keys():  # The same geocoding request will not be made twice
                    continue
                else:
                    try:
                        places_and_points[f"{place}"] = geocoder.geocode(place, timeout=120)[-1]
                    except geocoder.geocode(place, timeout=120) is None:
                        logger.error(f"Failed to geocode {place}")
                        places_and_points[f"{place}"] = (0, 0)

            return places_and_points

        nominatim_results = _trigger_geocoder(
            geocoder=Nominatim(user_agent=config.email),
            place_names=self.place_names
        )
        final_places_and_points = _trigger_geocoder(
            geocoder=Photon(),
            place_names=[key for key, value in nominatim_results if value == (0, 0)]
        )
        return final_places_and_points


def add_avg_trips_last_4_weeks(features: pd.DataFrame) -> pd.DataFrame:
    """
    Include a column for the average number of trips in the past 4 weeks.

    Returns:
        pd.DataFrame: the modified dataframe
    """
    if "average_trips_last_4_weeks" not in features.columns:
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        averages = 0.25 * (
                features[f"trips_previous_{1 * 7 * 24}_hour"] +
                features[f"trips_previous_{2 * 7 * 24}_hour"] +
                features[f"trips_previous_{3 * 7 * 24}_hour"] +
                features[f"trips_previous_{4 * 7 * 24}_hour"]
        )

        features.insert(
            loc=features.shape[1],
            column="average_trips_last_4_weeks",
            value=averages
        )

    return features


def add_hours_and_days(features: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Create features which consist of the hours and days of the week on which the
    departure or arrival is taking place.

    Returns:
        pd.DataFrame: the data frame with these features included
    """

    # The values in this column change types when uploading to Hopsworks, so this avoids an attribute 
    # errors during feature engineering.
    features[f"{scenario}_hour"] = pd.to_datetime(features[f"{scenario}_hour"], errors="coerce")

    times_and_entries = {
        "hour": features[f"{scenario}_hour"].apply(lambda x: x.hour),
        "day_of_the_week": features[f"{scenario}_hour"].apply(lambda x: x.dayofweek)
    }

    for time in times_and_entries.keys():
        features.insert(
            loc=features.shape[1],
            column=time,
            value=times_and_entries[time]
        )

    return features.drop(f"{scenario}_hour", axis=1)


def add_coordinates_to_dataframe(features: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    After forming the dictionary of places and coordinates, this function isolates
    the latitudes, and longitudes and places them in appropriately named columns of
    a target dataframe.
    """
    geodata = GeoData(data=features, scenario=scenario)
    places_and_points = geodata.geocode()

    for place in geodata.place_names:
        if place in places_and_points.keys():
            geodata.latitudes.append(places_and_points[place][0])
            geodata.longitudes.append(places_and_points[place][0])

    features[f"{scenario}_latitude"] = pd.Series(geodata.latitudes)
    features[f"{scenario}_longitude"] = pd.Series(geodata.longitudes)

    return features


def perform_feature_engineering(features: pd.DataFrame, scenario: str, geocode: bool) -> pd.DataFrame:
    """
    Initiate a chain of events that results in the accomplishment of the above feature
    engineering steps.

    Args:
        features: the features of our dataset
        scenario: whether we are looking at the starts or the ends of trips (enter "start" or "end")
        geocode: whether we want to initiate the geocoding procedures.

    Returns:
        pd.DataFrame: a dataframe containing the pre-existing features as well as the new ones.
    """
    logger.warning(f"Initiating feature engineering on the {scenario} data...")
    features_with_hours_and_days = add_hours_and_days(features=features, scenario=scenario)
    final_features = add_avg_trips_last_4_weeks(features=features_with_hours_and_days)

    assert "day_of_the_week" and "average_trips_last_4_weeks" in final_features.columns
    assert final_features["day_of_the_week"].isna().sum() == 0 and \
           final_features["average_trips_last_4_weeks"].isna().sum() == 0

    if geocode:
        final_features = add_coordinates_to_dataframe(features=features, scenario=scenario)
    return final_features
