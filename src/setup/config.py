import pandas as pd 

from datetime import datetime, UTC
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR


load_dotenv(PARENT_DIR / ".env")


class GeneralConfig(BaseSettings):
    _ = SettingsConfigDict(env_file=f"{PARENT_DIR}/.env", env_file_encoding="utf-8", extra="allow")

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
    feature_group_version: int = 1
    feature_view_version: int = 1

    # PostgreSQL
    database_public_url: str

    backfill_days: int = 210 
    current_hour: datetime = pd.to_datetime(datetime.now(tz=UTC)).floor("H")
    displayed_scenario_names: dict[str, str] = {"start": "Departures", "end": "Arrivals"} 


config = GeneralConfig()
