from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd 

import plotly.express as plotx


def plot_one_sample(
    example_station: int, 
    features: pd.DataFrame,
    scenario: str,
    targets: Optional[pd.Series] = None, 
    predictions: Optional[pd.Series] = None,
    display_title: Optional[bool] = True
):

    """Credit to Pau Labarta Bajo"""

    features_ = features.iloc[example_station]

    if targets is not None:

        target_ = targets.iloc[example_station]

    else:
        target_ = None

    columns = [column for column in features.columns if column.startswith("trips_previous_")]
    values = [features[column] for column in columns] + [target_]

    dates = pd.date_range(
        features_[f"{scenario}_hour"] - timedelta(hours=len(columns)),
        features_[f"{scenario}_hour"], freq="H"
    )

    if display_title:

        title = f'{scenario}_hour = {features_[{scenario}]}_hour, station_id = {features_[f"{scenario}_station_id"]}'

        fig = plotx.line(
            x=dates, y=values, 
            template="plotly_dark",
            markers=True, title=title
        )
    
    # Plot actual values if available
    if targets is not None:

        fig.add_scatter(
            x=dates[-1:], y = [target_],
            line_color="green", mode="markers",
            marker_size=10, name="Actual Number of Rides"
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
