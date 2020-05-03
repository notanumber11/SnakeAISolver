import sys
print("********************************")
print("********************************")
print("Starting Genetic algorithm...")
print(sys.version)
print("********************************")
print("********************************")


from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    # Check also to not modify the last layer
    ga = GeneticAlgorithm([9, 125, 1])
    games_to_play = 1
    population_size = 10
    selection_threshold = 0.1
    mutation_rate = 0.01
    iterations = 5
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play)
