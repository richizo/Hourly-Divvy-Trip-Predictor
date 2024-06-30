import hsfs
import hopsworks

from loguru import logger
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from src.setup.config import settings


def login_to_hopsworks() -> any:
    project = hopsworks.login(project=settings.comet_project_name, api_key_value=settings.hopsworks_api_key)
    return project


def get_feature_store() -> FeatureStore:
    """
    Login to Hopsworks and return a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = login_to_hopsworks()
    return project.get_feature_store()


def get_or_create_feature_group(scenario: str) -> FeatureGroup:
    feature_store = get_feature_store()
    feature_group = feature_store.get_or_create_feature_group(
        name=f"{scenario}_features",
        version=settings.feature_group_version,
        primary_key=[f"{scenario}_hour", f"{scenario}_station_id"]
    )
    return feature_group


def get_or_create_feature_view(scenario: str) -> FeatureView:
    feature_store = get_feature_store()
    feature_group = get_or_create_feature_group(scenario=scenario)

    try:
        feature_view = feature_store.create_feature_view(
            name=f"{scenario}_feature_view",
            version=settings.feature_view_version,
            query=feature_group.select_all()
        )

    except Exception as error:
        logger.exception(error)
        feature_view = feature_store.get_feature_view(
            name=f"{scenario}_feature_view",
            version=settings.feature_view_version
        )
    return feature_view
