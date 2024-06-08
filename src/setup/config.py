from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")
  
  # CometML
  comet_api_key: str
  comet_workspace: str
  comet_project_name: str
  
  # Names
  email: str

  # Numbers
  year: int
  
  
settings = Settings()
