import os
import json


def _is_aws() -> bool:
    running_in_aws = True
    operative_system = os.getenv('OS')
    if operative_system is not None and "windows" in operative_system.lower():
        running_in_aws = False
    print("Running in aws={}".format(running_in_aws))
    return running_in_aws


def _get_hyperparameters(path: str) -> str:
    with open(path) as f:
        print("********************************")
        print("The hyperparameters read from {} are:".format(path))
        data = json.load(f)
        print(data)
        print("********************************")
        return data


def get_hyperparameters():
    if _is_aws():
        return _get_hyperparameters("/opt/ml/input/config/hyperparameters.json")
    else:
        return _get_hyperparameters("hyperparameters.json")


def get_training_basic_genetic_output_folder() -> str:
    if _is_aws():
        return "/opt/ml/model/"
    else:
        return "C:\\Users\\Denis\\Desktop\\SnakePython\\data\\basic_genetic\\"
