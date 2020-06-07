from typing import List

from solvers.training_data_generators.classification.short_vision import ShortVision
from solvers.genetic.genetic_algorithm import GeneticAlgorithm
from solvers.training_data_generators.classification.distance_vision import DistanceVision
from utils import aws_snake_utils


def train_basic_genetic(checkpoint_path: str = None):
    ga = GeneticAlgorithm([5, 125, 4], ShortVision())
    _start_training(ga, checkpoint_path)


def train_advance_genetic(checkpoint_path: str = None):
    ga = GeneticAlgorithm([28, 20, 12, 4], DistanceVision())
    _start_training(ga, checkpoint_path)


def _start_training(ga, checkpoint_path):
    hyperparameters = aws_snake_utils.get_hyperparameters()
    ga.run(hyperparameters, checkpoint_path)
