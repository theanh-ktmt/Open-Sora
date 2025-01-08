import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
from loguru import logger


class MLFlowManager:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, exp_name: str):
        if not hasattr(self, "initialized") or not self.initialized:
            self.initialized = True
            self.exp_name = exp_name

    def setup_experiment(self):
        # set mlflow tracking URI
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
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

    def start_run(self, config: Optional[Dict[str, Any]]):
        # setup experiment
        self.setup_experiment()

        # start run
        now = datetime.now()
        run_name = f"run_{now.strftime('%Y%m%d%H%M%S')}"
        mlflow.start_run(run_name=run_name, experiment_id=self.exp_id)

        # Log config
        mlflow.log_dict(config, "config/config.yaml")

        # Log env vars
        mlflow.log_dict(dict(os.environ), "config/env_vars.json")

        # Log requirements
        mlflow.log_text(get_requirement_list(), "config/requirements.txt")

        # Log commit hash
        mlflow.log_param("commit_hash", get_current_commit_hash())

    def end_run(self):
        mlflow.end_run()


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
