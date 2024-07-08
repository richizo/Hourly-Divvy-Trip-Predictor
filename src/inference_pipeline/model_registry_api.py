import pickle
from loguru import logger

from sklearn.pipeline import Pipeline
from comet_ml import Experiment, ExistingExperiment, get_global_experiment, API

from src.setup.config import config
from src.setup.paths import MODELS_DIR
from src.training_pipeline.models import load_local_model


class ModelRegistryAPI:
    def __init__(self, model_name: str, model: Pipeline) -> None:
        self.model_name = model_name
        self.model = model

    def _load_local_model(self):
        with open(MODELS_DIR/self.model_name, "wb") as file:
            model = pickle.load(file)
        return model

    def get_latest_model_version(self, status: str) -> str:
        """
        Get the latest version of the requested model.
        Args:
            status: the registered status of the model on CometML

        Returns:
            int: the version of the latest model
        """
        api = API(api_key=config.comet_api_key)
        model_details = api.get_registry_model_details(
            workspace=config.comet_workspace, registry_name=self.model_name)["versions"]

        model_versions = [detail["version"]for detail in model_details if detail["status"] == status]
        return str(max(model_versions))

    def push_model_to_registry(self, status: str):
        """
        Find the model (saved locally), log it to CometML, and register it at the model registry.
        Args:
            status: the status that we want to give to the model during registration.

        Returns:
            None
        """
        model_file = MODELS_DIR/f"{self.model_name}.pkl"
        with open(model_file, "wb") as file:
            pickle.dump(self.model, file)

        stale_experiment = get_global_experiment()
        experiment = ExistingExperiment(api_key=stale_experiment.api_key, experiment_key=stale_experiment.id)

        logger.info("Logging model to Comet ML")
        experiment.log_model(name=self.model_name, file_or_folder=str(model_file))
        logger.success(f"Finished logging the {self.model_name} model")

        logger.success(f'Pushing model to the registry under {status.title()}')
        experiment.register_model(model_name=self.model_name, status=status)

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
            registry_name=self.model_name,
            version=self.get_latest_model_version(status=status),
            output_path=MODELS_DIR,
            expand=unzip,
            stage=status
        )

        model = load_local_model(model_name=self.model_name)
        return model
