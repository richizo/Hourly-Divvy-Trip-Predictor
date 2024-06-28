import hsfs
import hopsworks

from loguru import logger
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup

from src.setup.config import config


def get_feature_store() -> FeatureStore:
    """
    Login to Hopsworks and return a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(project=config.comet_project_name, api_key_value=config.comet_api_key)
    return project.get_feature_store()


#def get_or_create_feature_store(feature_group_config: FeatureGroupConfig) -> FeatureGroup: 