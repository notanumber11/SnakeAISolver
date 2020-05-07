import csv
import tensorflow as tf
from typing import List

import solvers.training.basic_training_data_generator


def create_csv(labels: List[str], data: List, name: str) -> None:
    """
    Creates a csv file with the name name_x being x the number of samples
    """
    path = "{}{}_samples_{}.csv".format(solvers.training.basic_training_data_generator.DATA_DIR, name, len(data))
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows([labels])
        writer.writerows(data)


def normalize_rad_angle(val):
    min = 0
    max = 6.28
    return (val - min) / (max - min)


def load_model(path: str):
    path = path.replace("/", "\\")
    new_model = tf.keras.models.load_model(path)
    return new_model
