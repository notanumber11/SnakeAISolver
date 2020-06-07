import csv
import os
from typing import List

import tensorflow as tf
from utils.snake_logger import get_module_logger

LOGGER = get_module_logger(__name__)


def create_csv(data_dir, labels: List[str], data: List, name: str) -> None:
    """
    Creates a csv file with the name name_x being x the number of samples
    """
    path = "{}{}_samples_{}.csv".format(data_dir, name, len(data))
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows([labels])
        writer.writerows(data)


def normalize_rad_angle(val):
    min = 0
    max = 6.28
    return (val - min) / (max - min)


def load_model(path: str):
    path = os.path.normpath(path)
    new_model = tf.keras.models.load_model(path)
    return new_model


def save_model(model, folder_path: str, file_name):
    folder_path = os.path.normpath(folder_path)
    full_file_path = os.path.join(folder_path, file_name)
    model.save(full_file_path)
    return full_file_path
