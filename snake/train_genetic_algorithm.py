from solvers.advance_genetic.advance_genetic_algorithm import AdvanceGeneticAlgorithm
from utils import aws_snake_utils
from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm

def train_basic_genetic():
    ga = GeneticAlgorithm([9, 125, 1])
    _start_training(ga)


def train_advance_genetic():
    ga = AdvanceGeneticAlgorithm([28, 20, 12, 4])
    _start_training(ga)


def _start_training(ga):
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
           game_size=game_size)
