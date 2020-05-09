import os
import sys
from utils import aws_snake_utils

print("*******************************************")
print("Starting to run train_advance_genetic_algorithm...")
print(sys.version)
print("The current directory is: {}".format(os.getcwd()))
print("The running environment is: {}".format(aws_snake_utils.get_running_environment()))
print("*******************************************")

from solvers.advance_genetic.advance_genetic_algorithm import AdvanceGeneticAlgorithm

if __name__ == '__main__':
    hyperparameters = aws_snake_utils.get_hyperparameters()
    ga = AdvanceGeneticAlgorithm([28, 20, 12, 1])
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
           game_size=10)
