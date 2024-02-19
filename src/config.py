from loguru import logger 
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

from sqlalchemy.engine import create_engine

class Settings(BaseSettings):
  
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")
  
  # Directories
  src_dir: DirectoryPath
  
  # CometML
  comet_api_key: str
  comet_workspace: str
  comet_project_name: str
  
  # Names
  email: str
  
  
settings = Settings()
logger.add("app.log", rotation="1 day", retention="12 hours", compression="zip")

engine = create_engine(url='sqlite:///db.sqlite')
