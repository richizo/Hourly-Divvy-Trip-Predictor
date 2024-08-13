import numpy as np

import streamlit as st

from tqdm import tqdm
from loguru import logger

import pandas as pd
from geopandas import GeoDataFrame

from shapely.geometry import Point 
from folium import Map, CircleMarker
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from src.setup.paths import GEOGRAPHICAL_DATA, INFERENCE_DATA, INDEXER_ONE, INDEXER_TWO


class MapMaker:

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario

    def scatterplot(self, geodata: GeoDataFrame, predictions: pd.DataFrame):
        """
        
        Args:
            scenario (str): _description_
            geojson (dict): _description_
            geodata (GeoDataFrame): _description_
            predictions (pd.DataFrame): _description_
        """
        with st.spinner("Building map..."):
            centre = [41.872866, -87.63363]
            station_names = geodata.iloc[:, 0].values
            station_ids = geodata.iloc[:, 1].values

            latitudes = geodata["latitudes"].values
            longitudes = geodata["longitudes"].values

            #rows_to_iterate = tqdm(iterable=range(geodata.shape[0]), desc="Gathering elements to display")
            #predictions_per_id = predictions.set_index(f"{scenario}_station_id")[f"predicted_{scenario}s"]

            marker_cluster = FastMarkerCluster(data=zip(latitudes, longitudes))
            folium_map = Map(location=centre)
            marker_cluster.add_to(parent=folium_map)

            displayed_map = st_folium(fig=folium_map, width=900, height=650)

        logger.success("Displayed map")
        st.sidebar.write("✅ Map Drawn")
        progress_bar.progress(4/n_steps)


