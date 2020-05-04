import sys

print("********************************")
print("Starting to run train_genetic_algorithm...")
print(sys.version)
print("********************************")

from utils import aws_snake_utils
from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    hyperparameters = aws_snake_utils.get_hyperparameters()
    ga = GeneticAlgorithm([9, 125, 1])
    games_to_play = int(hyperparameters.get("games_to_play", 1))
    population_size =  int(hyperparameters.get("population_size", 1000))
    selection_threshold = float(hyperparameters.get("selection_threshold", 0.1))
    mutation_rate = float(hyperparameters.get("mutation_rate", 0.01))
    iterations =  int(hyperparameters.get("iterations", 100))
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play)
