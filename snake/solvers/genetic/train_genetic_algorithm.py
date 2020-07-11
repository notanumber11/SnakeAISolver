from solvers.genetic.genetic_algorithm import GeneticAlgorithm
from solvers.genetic.hyperparameters import VALID_TRAINING_DATA_GENERATORS
from solvers.training_data_generators.classification.binary_vision import BinaryVision
from solvers.training_data_generators.classification.distance_vision import DistanceVision
from solvers.training_data_generators.classification.short_vision import ShortVision
from utils import aws_snake_utils

VALID_TYPES = ["short_vision", "binary_vision", "distance_vision"]


def train_genetic(checkpoint_path: str = None):
    h = aws_snake_utils.get_hyperparameters()
    training_data = h.training_data
    # Last layer has always 4 options, one per direction
    layers = h.layers + [4]
    if training_data == "short_vision":
        ga = GeneticAlgorithm(layers, ShortVision())
    elif training_data == "binary_vision":
        ga = GeneticAlgorithm(layers, BinaryVision())
    elif training_data == "distance_vision":
        ga = GeneticAlgorithm(layers, DistanceVision())
    else:
        raise ValueError("The training data={} is not valid. The valid training data are: {}".format(training_data,
                                                                                                     VALID_TRAINING_DATA_GENERATORS))
    ga.run(h, checkpoint_path)
