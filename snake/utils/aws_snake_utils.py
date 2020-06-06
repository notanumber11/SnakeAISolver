import json
import os
from typing import Dict


def _is_aws() -> bool:
    running_in_aws = True
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
    training_job_name_env = os.getenv('TRAINING_JOB_NAME')
    if training_job_name_env is None:
        running_in_aws = False
    return running_in_aws


def get_running_environment():
    if _is_aws():
        return "Running container in AWS sagemaker"
    if is_local_run():
        return "Running code in local run"
    if _is_container_not_in_aws():
        return "Running container in Linux"
    raise ValueError("Unknown running environment")


def _get_hyperparameters(path: str) -> Dict:
    print("The hyperparameters read from {} are:".format(path))
    with open(path) as f:
        data = json.load(f)
    print("Hyperparameters: {}".format(data))
    return data


def is_local_run():
    env_path = os.getenv("Path") if os.getenv("Path") is not None else os.getenv("PATH")
    if env_path is not None and "denis" in env_path.lower():
        return True
    return False


def _is_container_not_in_aws():
    return not _is_aws() and not is_local_run()


def get_hyperparameters() -> Dict:
    if is_local_run():
        return _get_hyperparameters("hyperparameters.json")
    elif _is_aws():
        return _get_hyperparameters("/opt/ml/input/config/3_result.json")
    elif _is_container_not_in_aws():
        return _get_hyperparameters("/opt/ml/code/3_result.json")
    raise ValueError("Could not find valid path for 3_result.json")


def get_training_output_folder() -> str:
    if is_local_run():
        return "models/experiments/"
    elif _is_aws():
        return "/opt/ml/model/"
    elif _is_container_not_in_aws():
        return "/opt/ml/code/"
    raise ValueError("Could not find valid path for training_data_generators output folder")
