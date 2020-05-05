import os
import sys
from utils import aws_snake_utils

print("*******************************************")
print("Starting to run train_genetic_algorithm...")
print(sys.version)
print("The current directory is: {}".format(os.getcwd()))
print("The running environment is: {}".format(aws_snake_utils.get_running_environment()))
print("*******************************************")

from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    hyperparameters = aws_snake_utils.get_hyperparameters()
    ga = GeneticAlgorithm([9, 125, 1])
    games_to_play = int(hyperparameters["games_to_play"])
    population_size =  int(hyperparameters["population_size"])
    selection_threshold = float(hyperparameters["selection_threshold"])
    mutation_rate = float(hyperparameters["mutation_rate"])
    iterations =  int(hyperparameters["iterations"])
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play)
