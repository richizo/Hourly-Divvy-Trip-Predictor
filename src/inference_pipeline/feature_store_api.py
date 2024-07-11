import hopsworks

from loguru import logger
from hsfs.feature_view import FeatureView
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup


class FeatureStoreAPI:
    def __init__(
            self,
            api_key: str,
            scenario: str,
            project_name: str,
            primary_key: list[str],
            event_time: str
    ) -> None:
        self.api_key = api_key
        self.scenario = scenario
        self.project_name = project_name

        self.event_time = event_time
        self.primary_key = primary_key

    def login_to_hopsworks(self) -> any:
        project = hopsworks.login(project=self.project_name, api_key_value=self.api_key)
        return project

    def get_feature_store(self) -> FeatureStore:
        """
        Login to Hopsworks and return a pointer to the feature store

        Returns:
            FeatureStore: pointer to the feature store
        """
        project = self.login_to_hopsworks()
        return project.get_feature_store()

    def get_or_create_feature_group(
            self,
            name: str,
            version: int,
            description: str
    ) -> FeatureGroup:
        """
        Create or connect to a feature group with the specified name, and 
        return an object that represents it.

        Returns:
            FeatureGroup: a representation of the fetched or created feature group
        """
        feature_store = self.get_feature_store()
        feature_group = feature_store.get_or_create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=self.primary_key,
            event_time=self.event_time
        )
        return feature_group

    def get_or_create_feature_view(self, name: str, version: int, feature_group: FeatureGroup) -> FeatureView:
        """

        Args:
            name: the name of the feature view to fetch or create
            version: the version of the feature view
            feature_group: the feature group object that will be queried if the feature view needs to be created

        Returns:

        """
        feature_store = self.get_feature_store()

        try:
            feature_view = feature_store.create_feature_view(
                name=name,
                version=version,
                query=feature_group.select_all()
            )

        except Exception as error:
            logger.exception(error)
            feature_view = feature_store.get_feature_view(name=name, version=version)
        return feature_view
