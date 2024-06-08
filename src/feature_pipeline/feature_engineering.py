import pandas as pd 

from loguru import logger
from warnings import simplefilter

from geopy.geocoders import Nominatim, Photon
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.setup.config import settings


def perform_feature_engineering(data: pd.DataFrame, scenario: str, geocode: bool):
    feature_engineer = FeatureEngineering(data=data, scenario=scenario)
    return feature_engineer._get_pipeline(geocode=geocode)
    

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, data: pd.DataFrame, scenario: str, y=None) -> None:

        self.data = data
        self.y = y
        self.scenario = scenario
        self.added_averages = 0.25*(
            self.data[f"trips_previous_{1*7*24}_hour"] + self.data[f"trips_previous_{2*7*24}_hour"] + \
            self.data[f"trips_previous_{3*7*24}_hour"] + self.data[f"trips_previous_{4*7*24}_hour"]
        )

        self.times_and_entries = {
            "hour": self.data[f"{self.scenario}_hour"].dt.hour, 
            "day_of_the_week": self.data[f"{self.scenario}_hour"].dt.dayofweek
        }
    
    def _fit(self):
        return self 

    def _add_avg_trips_last_4_weeks(self) -> pd.DataFrame:
        """
        Include a column for the average number of trips in the past 4 weeks.

        Returns:
            pd.DataFrame: the modified dataframe
        """
        if "average_trips_last_4_weeks" not in self.data.columns:
            simplefilter(action = "ignore", category=pd.errors.PerformanceWarning)
            self.data.insert(
                loc=self.data.shape[1], 
                column="average_trips_last_4_weeks", 
                value=self.added_averages, 
                allow_duplicates=False
            )
        return self.data

    def _add_hours_and_days(self) -> pd.DataFrame:
        """
        Create features which consist of the hours and days of the week on which the 
        departure or arrival is taking place.

        Returns:
            pd.DataFrame: the data frame with these features included
        """
        
        for time in self.times_and_entries.keys():
            self.data.insert(
                loc=self.data.shape[1], 
                column=time, 
                value=self.times_and_entries[time]
            )   

        return self.data.drop(f"{self.scenario}_hour", axis = 1)

    def _get_pipeline(self, geocode: bool) -> Pipeline:
        
        steps = [
            FunctionTransformer(func = self._add_avg_trips_last_4_weeks, validate = False),
            FunctionTransformer(func=self._add_hours_and_days)
        ]

        if geocode:
            steps.append(
                FunctionTransformer(func=self._add_coordinates_to_dataframe)
            )

        return make_pipeline(*steps)

    def _add_coordinates_to_dataframe(self) -> None:
        """
        After forming the dictionary of places and coordinates, this function isolates
        the latitudes, and longitudes and places them in appropriately named columns of 
        a target dataframe.
        """
        geodata = GeoData(data = self.data, scenario=self.scenario)
        places_and_points = geodata._geocode()

        for place in geodata.place_names:
            if place in places_and_points.keys():
                geodata.latitudes.append(places_and_points[place][0])
                geodata.longitudes.append(places_and_points[place][0])

        self.data[f"{self.scenario}_latitude"] = pd.Series(geodata.latitudes)
        self.data[f"{self.scenario}_longitude"] = pd.Series(geodata.longitudes)


class GeoData():

    def __init__(self, data: pd.DataFrame, scenario: str) -> None:
        self.data = data
        self.scenario = scenario
        self.latitudes, self.longitudes = []
        self.place_names = self.data[f"{scenario}_station_name"].unique().to_list()
                      
    def _geocode(self) -> dict:
        """
        Initialises the Nominatim geocoder, and applies it to a list of place names. 
        It then generates the coordinates of each place and creates key, value pairs 
        of each place name with its respective coordinate. 

        Some of the locations will not be successfully processed by the geocoder, 
        causing it (the geocoder) to return None. For each such case, the function 
        provides (0,0) as the value corresponding to the place name. After this,
        the locations that were unable to be geocoded by Nominatim will be run 
        through the Photon geocoder. This will (hopefully) result in the geocoding
        many of these locations. Those that are unsuccesfully geocoded will again 
        have (0,0) as their corresponding coordinates.

        Args:
            place_names (list): the names of the places that are to be 
                                geocoded.

        Returns:
            dict: a dictionary which contains key value pairs of place names and 
                  coordinates
        """
        def __trigger_geocoder(self, geocoder: Nominatim|Photon, place_names: list) -> dict:

            places_and_points = {}
            for place in place_names:
                if place in places_and_points.keys(): # The same geocoding request will not be made twice
                    continue
                elif geocoder.geocode(place, timeout=120) is not None:
                    places_and_points[f"{place}"] = places_and_points.geocode(place, timeout=None)[-1]
                else:
                    logger.error(f"Failed to geocode {place}")
                    places_and_points[f"{place}"] = (0,0)
            return places_and_points

        nominatim_results = __trigger_geocoder(
            geocoder=Nominatim(user_agent=settings.email), 
            place_names = self.place_names
        )

        final_places_and_points = __trigger_geocoder(
            geocoder=Photon(),
            place_names = [key for key, value in nominatim_results if value == (0,0)]
        )

        return final_places_and_points




