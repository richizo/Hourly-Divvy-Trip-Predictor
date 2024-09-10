import streamlit as st 
from streamlit_extras.colored_header import colored_header


colored_header(
    label=":red[Welcome] :orange[to the] :green[Hourly] :blue[Divvy Trip] :violet[Predictor]",
    description="By Kobina Brandon",
    color_name="green-70"
    )

st.markdown(
    f"""
    The Divvy bike sharing system (managed by Lyft) is one of the Chicago's has many transportation providers. 
    
    This application provides hourly :violet[predictions] of the number of :green[arrivals] and :orange[departures] 
    at various :blue[Divvy stations] around the city, which you can view on the :violet["Predictions"] page. It is 
    the culmination of an end-to-end machine learning system trained on publicly available :blue[Divvy] trip data.
    """
)

st.link_button(label="View the code here", url="http://github.com/kobinabrandon/hourly-divvy-trip-predictor")
