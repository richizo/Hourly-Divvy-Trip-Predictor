import streamlit as st 


st.write("# Welcome to the Hourly Divvy Trip Predictor Service")

st.markdown(
    """
    This service provides predictions of the number of arrivals and departures at various Divvy stations around the
    city of Chicago. Divvy have generously made their trip data publicly available, and I've used it to build an 
    end-to-end machine learning system that provides this service.

    Take a look at the "Predictions" page, where we provide the main results of the model: the predicted number of 
    arrivals and departures per hour at various Divvy stations in the city.
    """
)
