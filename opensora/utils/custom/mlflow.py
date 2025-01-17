import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Union

import mlflow
from loguru import logger

from opensora.utils.custom.config import ConfigurationManager


class MLFlowManager:
    __instance = None
    __active = ConfigurationManager.get("ENABLE_MLFLOW")

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, exp_name: str):
        if not hasattr(self, "initialized") or not self.initialized:
            self.initialized = True
            self.exp_name = exp_name

    def setup_experiment(self) -> None:
        """Setup experiment information. Setup tracking URI and creating experiment."""
        if not self.__active:
            return

        # set mlflow tracking URI
        mlflow_tracking_uri = ConfigurationManager.get("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

        # create experiment if needed
        try:
            self.exp_id = mlflow.create_experiment(self.exp_name)
            logger.info(f"Experiment '{self.exp_name}' created with ID: {self.exp_id}")
        except mlflow.exceptions.MlflowException:
            # If the experiment already exists, get its ID
            experiment = mlflow.get_experiment_by_name(self.exp_name)
            self.exp_id = experiment.experiment_id

    def start_run(self, config: Optional[Dict[str, Any]]) -> None:
        """Start a new run."""
        if not self.__active:
            return

        # setup experiment
        self.setup_experiment()

        # start run
        now = datetime.now()
        run_name = f"run_{now.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name, experiment_id=self.exp_id)

        # Log config
        mlflow.log_dict(config, "config/config.yaml")

        # Log settings
        mlflow.log_dict(ConfigurationManager.CONFIGS, "config/settings.json")

        # Log env vars
        mlflow.log_dict(dict(os.environ), "config/env_vars.json")

        # Log requirements
        mlflow.log_text(get_requirement_list(), "config/requirements.txt")

        # Log commit hash
        mlflow.log_param("commit_hash", get_current_commit_hash())

    def end_run(self) -> None:
        """End current run."""
        if not self.__active:
            return
        mlflow.end_run()

    @classmethod
    def set_tag(cls, tag: str, value: Optional[str] = None) -> None:
        if not cls.__active:
            return
        if value:
            mlflow.set_tag(tag, value)
        else:
            mlflow.set_tag(tag, "True")

    @classmethod
    def log_params(cls, params: Dict[str, Any]) -> None:
        if not cls.__active:
            return
        mlflow.log_params(params)

    @classmethod
    def log_metric(cls, key: str, value: float, step: int = 0) -> None:
        if not cls.__active:
            return
        mlflow.log_metric(key, value, step)

    @classmethod
    def log_metrics(cls, metrics: Dict[str, Any], step: int = 0) -> None:
        if not cls.__active:
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step)

    @classmethod
    def log_artifact(cls, path: str, dest_dir: str) -> None:
        if not cls.__active:
            return

        if os.path.isdir(path):
            mlflow.log_artifacts(path, dest_dir)
        elif os.path.isfile(path):
            mlflow.log_artifact(path, dest_dir)
        else:
            raise NotImplementedError("Path {} is neither file nor directory.".format(path))

    @classmethod
    def log_file(cls, input: Union[str, dict], filepath: str) -> None:
        if not cls.__active:
            return

        if isinstance(input, str):
            mlflow.log_text(input, filepath)
        elif isinstance(input, dict):
            mlflow.log_dict(input, filepath)
        else:
            raise NotImplementedError("Unsupported dtype: {}".format(type(input)))


def get_requirement_list() -> str:
    """Return list of requirements in the string format of requirements.txt file"""
    result = subprocess.run(["pip", "list", "--format=freeze"], stdout=subprocess.PIPE, text=True, check=True)
    requirements_str = result.stdout
    return requirements_str


def get_current_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    commit_hash = result.stdout.strip()
    return commit_hash
