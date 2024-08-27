import pandas as pd 

from datetime import datetime, UTC
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR


load_dotenv(PARENT_DIR / ".env")


class GeneralConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=f"{PARENT_DIR}/.env", env_file_encoding="utf-8", extra="allow")

    # Names 
    year: int = 2024
    n_features: int = 672
    email: str

    # CometML
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    # Hopsworks
    hopsworks_api_key: str
    hopsworks_project_name: str
    feature_group_version: int = 6
    feature_view_version: int = 1

    current_hour: datetime = pd.to_datetime(datetime.now(UTC)).floor("H")
    displayed_scenario_names: dict = {"start": "Departures", "end": "Arrivals"} 


class FeatureGroupConfig(BaseSettings):
    name: str
    version: int
    primary_key: list[str]
    online_enabled: bool | None = False  # because this is a batch ML system. 
    event_time: str


class FeatureViewConfig(BaseSettings):
    name: str
    version: int
    feature_group: FeatureGroupConfig


config = GeneralConfig()
