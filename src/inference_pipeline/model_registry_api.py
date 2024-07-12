import pickle
from pathlib import Path

from loguru import logger
from sklearn.pipeline import Pipeline
from comet_ml import ExistingExperiment, get_global_experiment, API

from src.setup.config import config
from src.setup.paths import COMET_SAVE_DIR, LOCAL_SAVE_DIR
from src.training_pipeline.models import load_local_model


class ModelRegistry:
    def __init__(self, scenario: str, model_name: str, tuned_or_not: str, model: Pipeline) -> None:
        self.model = model
        self.scenario = scenario
        self.model_name = model_name
        self.tuned_or_not = tuned_or_not
        self.registered_name = self._set_registered_name()

    def _set_registered_name(self) -> str:
        return f"{self.model_name.title()} ({self.tuned_or_not} for {self.scenario}s)"

    def get_registered_model_versions(self, status: str) -> list:
        api = API(api_key=config.comet_api_key)
        model_details = api.get_registry_model_details(
            workspace=config.comet_workspace, registry_name=self.registered_name)["versions"]

        model_versions = [detail["version"]for detail in model_details if detail["status"] == status]
        return model_versions

    def get_latest_model_version(self, status: str) -> str:
        """
        Get the latest version of the requested model.
        Args:
            status: the registered status of the model on CometML

        Returns:
            int: the version of the latest model
        """
        return max(self.get_registered_model_versions(status=status))

    def push_model_to_registry(self, status: str, version: str) -> None:
        """
        Find the model (saved locally), log it to CometML, and register it at the model registry.

        Args:
            status: the status that we want to give to the model during registration.
            version: the version of the model being pushed

        Returns:
            None
        """
        stale_experiment = get_global_experiment()
        experiment = ExistingExperiment(api_key=stale_experiment.api_key, experiment_key=stale_experiment.id)

        logger.info("Logging model to Comet ML")
        model_file = LOCAL_SAVE_DIR/f"{self._set_registered_name()}.pkl"
        experiment.log_model(name=self._set_registered_name(), file_or_folder=str(model_file))
        logger.success(f"Finished logging the {self.model_name} model")

    #     if len(self.get_registered_model_versions(status=status)) != 0:
    #         logger.warning("There is a pre-existing model")
    #         latest_model_version = self.get_latest_model_version(status=status)
    #         if latest_model_version <= version:
    #             version = latest_model_version.replace(latest_model_version[-1], "2")

        logger.info(f'Pushing version {version} model to the registry under "{status.title()}"...')
        experiment.register_model(model_name=self._set_registered_name(), status=status, version=version)

    def download_latest_model(self, status: str, unzip: bool) -> Pipeline:
        """
        Download the latest version of the requested model to the MODEL_DIR directory,
        load the file using pickle, and return it.

        Args:
            status: the status of the requested model on CometML
            unzip: whether to unzip the downloaded zipfile.

        Returns:
            Pipeline: the original model file
        """
        api = API(api_key=config.comet_api_key)
        api.download_registry_model(
            workspace=config.comet_workspace,
            registry_name=self.registered_name,
            version=self.get_latest_model_version(status=status),
            output_path=COMET_SAVE_DIR,
            expand=unzip,
            stage=status
        )

        model = load_local_model(
            directory=COMET_SAVE_DIR,
            model_name=self.model_name,
            scenario=self.scenario,
            tuned_or_not=self.tuned_or_not
        )
        return model
