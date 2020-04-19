import csv
from typing import List

DATA_DIR = "C:/Users/Denis/Desktop/SnakePython/data/basic_dnn/"
TRAINING_DATA_BASIC_DNN = "training_data_basic_dnn"
MODEL_BASIC_DNN = "mode_basic_dnn"
LABELS = ["up", "down", "left", "right", "up available", "down available", "left available", "right available",
          "angle to apple", "reward"]


def create_csv(labels: List[str], data: List, name: str) -> None:
    """
    Creates a csv file with the name name_x being x the number of samples
    """
    path = "{}{}_samples_{}.csv".format(DATA_DIR, name, len(data))
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows([labels])
        writer.writerows(data)


def normalize_rad_angle(val):
    min = 0
    max = 6.28
    return (val - min) / (max - min)
