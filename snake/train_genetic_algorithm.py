import argparse
import sys

print("********************************")
print("********************************")
print("Starting train_genetic_algorithm...")
print(sys.version)
print("********************************")
print("********************************")

from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    print("Reading hyperparameters...")
    parser = argparse.ArgumentParser()
    # https://sagemaker.readthedocs.io/en/stable/overview.html
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--games_to_play', type=int, default=2)
    parser.add_argument('--population_size', type=int, default=1000)
    parser.add_argument('--selection_threshold', type=int, default=0.1)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--iterations', type=float, default=100)

    args, _ = parser.parse_known_args()

    # Check also to not modify the last layer
    ga = GeneticAlgorithm([9, 125, 1])
    games_to_play = args.games_to_play
    population_size = args.population_size
    selection_threshold = args.selection_threshold
    mutation_rate = args.mutation_rate
    iterations = args.iterations
    print("********************************")
    print("********************************")
    print("The hyperparameters are:")
    print(args)
    print("********************************")
    print("********************************")
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play)
