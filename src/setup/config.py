from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../../.env", env_file_encoding="utf-8", extra="allow")

    # CometML
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    # Names
    email: str
    scenario: str

    @field_validator("scenario")
    def validate_scenarios(self, scenario: str):
        assert scenario in ["start", "end"], f"Invalid value for scenario: {scenario}"
        return scenario


settings = Settings()
