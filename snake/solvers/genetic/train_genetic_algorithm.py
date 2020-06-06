from typing import List

from solvers.data_providers import dnn_training_data_generator
from solvers.genetic.genetic_algorithm import GeneticAlgorithm
from solvers.data_providers.distance_vision_training_data_generator import DistanceVisionTrainingDataGenerator
from utils import aws_snake_utils


def train_basic_genetic(model_paths: List[str] = None):
    ga = GeneticAlgorithm([5, 125, 4], dnn_training_data_generator)
    _start_training(ga, model_paths)


def train_advance_genetic(model_paths: List[str] = None):
    ga = GeneticAlgorithm([28, 20, 12, 4], DistanceVisionTrainingDataGenerator())
    _start_training(ga, model_paths)


def _start_training(ga, model_paths: List[str] = None):
    hyperparameters = aws_snake_utils.get_hyperparameters()
    games_to_play = int(hyperparameters["games_to_play"])
    population_size = int(hyperparameters["population_size"])
    selection_threshold = float(hyperparameters["selection_threshold"])
    mutation_rate = float(hyperparameters["mutation_rate"])
    iterations = int(hyperparameters["iterations"])
    game_size = int(hyperparameters["game_size"])
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play,
           game_size=game_size,
           model_paths=model_paths)
