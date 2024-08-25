"""
This experimental module contains code that displays the locations of the various 
stations using a map. It has been challenging to create an implementation of this that 
produces a good experience.
"""
import pandas as pd
import streamlit as st

from folium import Map
from geopandas import GeoDataFrame
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium

from src.inference_pipeline.frontend.main import ProgressTracker
from src.inference_pipeline.frontend.data import prepare_geodata_df
from src.inference_pipeline.frontend.data import load_geojson


@st.cache_resource
def make_scatterplot(geodata: pd.DataFrame | GeoDataFrame):
    """
    
    Args:
        geodata (GeoDataFrame): _description_
    """
    with st.spinner("Building map..."):
        centre = [41.872866, -87.63363]

        latitudes = geodata["latitudes"].values
        longitudes = geodata["longitudes"].values

        folium_map = Map(location=centre)
        marker_cluster = FastMarkerCluster(data=zip(latitudes, longitudes))
        marker_cluster.add_to(parent=folium_map)

        return folium_map  

            
if __name__ != "__main__":
    tracker = ProgressTracker(n_steps=3)
    geojson = load_geojson(scenario="start")
    tracker.next()  # Keeping the progress bar code outside the execution of these cached functions

    geodata = prepare_geodata_df(scenario="start", geojson=geojson)
    tracker.next()

    map_ = make_scatterplot(geodata=geodata)
    st_folium(fig=map_, width=900, height=650)
    st.sidebar.write("✅ Map Drawn")
    tracker.next()
