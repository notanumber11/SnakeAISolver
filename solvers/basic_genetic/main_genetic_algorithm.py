import sys

sys.path.extend(['C:\\Users\\Denis\\Desktop\\SnakePython', 'C:/Users/Denis/Desktop/SnakePython'])

import tensorflow as tf

from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm


def load_model(path: str):
    path = path.replace("/", "\\")
    new_model = tf.keras.models.load_model(path)
    return new_model


if __name__ == '__main__':
    # Check also to not modify the last layer
    ga = GeneticAlgorithm([9, 125, 1])
    games_to_play = 1
    population_size = 1000
    selection_threshold = 0.1
    mutation_rate = 0.01
    iterations = 50
    ga.run(population_size=population_size,
           selection_threshold=selection_threshold,
           mutation_rate=mutation_rate,
           iterations=iterations,
           games_to_play_per_individual=games_to_play)


