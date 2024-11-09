import streamlit as st 
from streamlit_extras.colored_header import colored_header


colored_header(
    label=":red[Welcome] :orange[to the] :blue[Hourly Divvy Trip] :violet[Predictor]",
    description="By Kobina Brandon",
    color_name="green-70"
)

st.markdown(
    """
    Chicago's Divvy bike-sharing system (operated by Lyft) is one of the largest of its kind in North America.

    This application provides hourly :violet[predictions] of the number of :green[arrivals] and :red[departures] 
    at various :blue[Divvy stations] around the city, which you can view on the :violet["Predictions"] page. It is 
    the culmination of my open source end-to-end machine learning system that I trained on publicly available :blue[Divvy] 
    trip data.
    """
)


colored_header(
    label=":red[What] :orange[is the] :blue[point of this service]?",
    description="To solve a business problem of course!",
    color_name="green-70"
)


st.markdown(
    """
    Having these hourly predictions provides the following benefits:
    
    - :blue[Proactive Resource Allocation:] access to such predictions enables proactive re-allocation of bikes in anticipation
    of sudden surges or reductions in demand. 

    - :blue[Long Term Operational Efficiency:] by monitoring the predictions produced by the service over time, management
    can determine bike demand at peak times. In other words, by studying trends in the predictions over time, management can
    rebalance the supply of bikes between stations, and ensure that bikes can be moved from stations that have a surplus of 
    bikes to those that are expected to have more departures. 

    - :blue[Strategic Planning:] by studying trends in the predictions over time, management can gain insight into usage 
    patterns across different times, days of the week, and seasons. This would enable long-term planning for the 
    installation of new stations, expansion of existing stations, and possible marketing iniatives tailored for customers 
    at various times and locations.
    
    - :blue[Improve Customer Satisfaction:] ensuring that the appropriate number of bikes are available at all times prevents
    customers from encountering empty stations, and from queueing to wait for new bikes to become available.

    - :blue[Dynamic Pricing/Incentives:] using predictions of bike demand, and known bike availability, management could 
    dynamically introduce promotions and pricing models designed to attract customers in certain areas who are travelling 
    to particular locations at certain times, thereby aiding bike re-allocation efforts without the deployment of staff 
    (and its associated costs), while increasing revenues.
    """
)