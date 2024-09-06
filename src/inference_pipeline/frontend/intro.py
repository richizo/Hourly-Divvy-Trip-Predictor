import streamlit as st 


st.header(":red[Welcome] :orange[to the] :blue[Divvy] :green[Trip] :violet[Predictor]")

st.markdown(
    """
    This service provides hourly predictions of the number of :green[arrivals] and :orange[departures] at various \ 
    :blue[Divvy stations] around the city of Chicago. Divvy have generously made their trip data publicly available, 
    and I've used it to build the end-to-end machine learning system that provides this service.

    Take a look at the "Predictions" page, where we provide the main results of the model.
    """
)


    