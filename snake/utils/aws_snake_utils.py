import os
import json


def _is_aws() -> bool:
    running_in_aws = True
    operative_system = os.getenv('OS')
    print("The env OS is: {}".format(operative_system))
    if operative_system is not None and "windows" in operative_system.lower():
        running_in_aws = False
    print("Running in aws={}".format(running_in_aws))
    try:
        # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
        aws_env = os.getenv('TRAINING_JOB_NAME')
        print("The env TRAINING_JOB_NAME is: {}".format(aws_env))
    except:
        print("Exception reading TRAINING_JOB_NAME")
    return running_in_aws


def _get_hyperparameters(path: str) -> str:
    print("The hyperparameters read from {} are:".format(path))
    with open(path) as f:
        data = json.load(f)
    print("Hyperparameters: {}".format(data))
    return data

def get_hyperparameters():
    path = "/opt/ml/input/config/hyperparameters.json" if _is_aws() else "hyperparameters.json"
    try:
        return _get_hyperparameters(path)
    except:
        # Use the defaults provided in train_basic_genetic_algorithm
        print("Could not read file from path: {}".format(path))
        return {}

def get_training_basic_genetic_output_folder() -> str:
    if _is_aws():
        return "/opt/ml/game/"
    else:
        return "..\\data\\basic_genetic\\"
