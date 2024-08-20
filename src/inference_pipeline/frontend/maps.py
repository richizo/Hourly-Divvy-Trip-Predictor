import numpy as np
import pandas as pd

import streamlit as st

from tqdm import tqdm
from loguru import logger

from geopandas import GeoDataFrame
from shapely.geometry import Point 
from folium import Map, CircleMarker
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from src.setup.config import config, choose_displayed_scenario_name
from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_TWO, INDEXER_TWO

from src.inference_pipeline.frontend.predictions import (
    tracker, load_geojson, make_geodataframe, get_features, get_hourly_predictions
)



@st.cache_data
def make_geodataframe(scenario: str, geojson: dict) -> GeoDataFrame:
    
    coordinates = []
    station_ids = []
    station_names = []
    
    for detail_index in range(len(geojson["features"])):
            
        detail: dict = geojson["features"][detail_index]

        coordinates.append(
            detail["geometry"]["coordinate"]
        )

        station_ids.append(
            detail["properties"]["station_id"]
        )

        station_names.append(
            detail["properties"]["station_name"]
        )
    
    latitudes = [point[1] for point in coordinates]
    longitudes = [point[0] for point in coordinates]

    data = pd.DataFrame(
        data={
            f"{scenario}_station_name": station_names,
            f"{scenario}_station_id": station_ids,
            "latitudes": latitudes,
            "longitudes": longitudes
        }
    )

    #  data["coordinates"] = data["coordinates"].apply(lambda x: Point(x[0], x[1]))
    #  geodata = GeoDataFrame(data=data, geometry="coordinates")

    #  geodata.set_crs(epsg=4326, inplace=True)

    return data

@st.cache_resource
def make_scatterplot(geodata: GeoDataFrame):
    """
    
    Args:
        scenario (str): _description_
        geojson (dict): _description_
        geodata (GeoDataFrame): _description_
    """
    with st.spinner("Building map..."):
        centre = [41.872866, -87.63363]
        station_names = geodata.iloc[:, 0].values
        station_ids = geodata.iloc[:, 1].values

        latitudes = geodata["latitudes"].values
        print(len(latitudes))
        longitudes = geodata["longitudes"].values

        #rows_to_iterate = tqdm(iterable=range(geodata.shape[0]), desc="Gathering elements to display")
        #predictions_per_id = predictions.set_index(f"{scenario}_station_id")[f"predicted_{scenario}s"]

        marker_cluster = FastMarkerCluster(data=zip(latitudes, longitudes))
        folium_map = Map(location=centre)
        marker_cluster.add_to(parent=folium_map)

        displayed_map = st_folium(fig=folium_map, width=900, height=650)

    st.sidebar.write("✅ Map Drawn")
    tracker.next()
    

for scenario in ["start", "end"]:
    displayed_scenario_names = choose_displayed_scenario_name()
    if displayed_scenario_names[scenario] in user_scenario_choice:

        # Prepare geodata
        geojson = load_geojson(scenario="start")
        geodata = make_geodataframe(scenario=scenario, geojson=geojson)
        tracker.next()
        
        make_scatterplot(geodata=geodata)
