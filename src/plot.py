import pandas as pd 
from datetime import timedelta

import plotly.express as plot


def plot_one_sample(
    example_station: int, 
    features: pd.DataFrame,
    scenario: str,
    targets: pd.Series = None, 
    predictions: pd.Series = None,
    display_title: bool = True
):
    """
  
    
    Credit to Pau Labarta Bajo
    """
    if targets is not None:
        target_ = targets.iloc[example_station]
    else:
        target_ = None


    features_ = features.iloc[example_station]
    columns = [column for column in features.columns if column.startswith("trips_previous_")]
    values = [features[column] for column in columns] + [target_]

    dates = pd.date_range(
        start=features_[f"{scenario}_hour"] - timedelta(hours=len(columns)),
        end=features_[f"{scenario}_hour"], 
        freq="h"
    )

    if display_title:

        #  title = f'{scenario}_hour = {features_[{scenario}]}_hour, station_id = {features_[f"{scenario}_station_id"]}'

        fig = plot.line(
            x=dates, 
            y=values, 
            template="plotly_dark",
            markers=True
        )
    
    # Plot actual values if available
    if targets is not None:

        fig.add_scatter(
            x=dates[-1:],
            y=[target_],
            mode="markers",
            marker_size=10, 
            line_color="green",
            name="Actual Number of Trips"
        )

    # Plot predicted values if available
    if predictions is not None:
        predictions_ = predictions.iloc[example_station]
        
        fig.add_scatter(
            x=dates[-1:], y=[predictions_],
            line_color="red", mode="markers",
            marker_symbol="x", marker_size=15,
            name="Predicted Values"
        )

    return fig
