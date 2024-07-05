from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR


class GeneralConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=f"{PARENT_DIR}/.env", env_file_encoding="utf-8", extra="allow")

    # CometML
    year: int
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    # Hopsworks
    hopsworks_api_key: str
    hopsworks_project_name: str
    feature_group_version: int
    feature_view_version: int

    # Names
    email: str


class FeatureGroupConfig(BaseSettings):
    name: str
    version: str
    primary_key: list[str]
    event_time: str
    online_enabled: bool | None = False  # because this is a batch ML system. 


class FeatureViewConfig(BaseSettings):
    name: str
    version: int
    feature_group: FeatureGroupConfig


config = GeneralConfig()
