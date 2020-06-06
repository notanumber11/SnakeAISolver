import gc
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game.game_seed_creator import create_random_game_seed
from solvers.data_providers import dnn_training_data_generator as training_generator, data_utils
from solvers.data_providers.data_utils import load_model
from solvers.genetic.crossover import crossover
from solvers.genetic.evaluation import evaluate_population, set_model_weights
from solvers.genetic.mutation import mutate
from solvers.genetic.selection import elitism_selection
from utils import aws_snake_utils
from utils.snake_logger import get_module_logger
from utils.timing import timeit

LOGGER = get_module_logger(__name__)


class GeneticAlgorithm:

    def __init__(self, layers_size: List[int], training_data_generator):
        self.layers_size = layers_size
        self.number_of_layers = len(self.layers_size)
        self.model = self.build_model()
        self.training_generator = training_data_generator
        if layers_size[-1] < 2:
            raise ValueError("AdvanceGeneticAlgorithm expects the output layer to be >1 since it uses classification {}"
                             .format(layers_size))

    def get_random_initial_population_genetic(self, population_size: int) -> List:
        population_genetic = []
        for i in range(population_size):
            # For each layer we need generate random weights
            model_genetic = []
            model_weights = self.model.get_weights()
            for j in range(len(model_weights)):
                weights = np.random.uniform(low=-1, high=1, size=model_weights[j].shape)
                model_genetic.append(weights)
            population_genetic.append(model_genetic)
        return population_genetic

    def run(self, population_size, selection_threshold, mutation_rate, iterations, games_to_play_per_individual=1,
            game_size=6, model_paths: List[str] = None):
        model_description = "pop={}_sel={}_mut_{}_it_{}_games_{}_game_size_{}/".format(population_size,
                                                                                       selection_threshold,
                                                                                       mutation_rate, iterations,
                                                                                       games_to_play_per_individual,
                                                                                       game_size)
        LOGGER.info("Running game: {}".format(model_description))
        dir_path = aws_snake_utils.get_training_output_folder() + model_description
        if model_paths is None:
            population_genetic = self.get_random_initial_population_genetic(population_size)
        else:
            population_genetic = self.load_initial_population_genetic(model_paths, population_size, mutation_rate,
                                                                      games_to_play_per_individual, game_size)
        # Iterate
        for i in range(iterations):
            snake_size = 4
            LOGGER.info("Running iteration: {}    with game_size: {}".format(i, game_size))
            new_population_genetic, population_evaluated = self.execute_iteration(population_genetic,
                                                                                  games_to_play_per_individual,
                                                                                  selection_threshold, mutation_rate,
                                                                                  population_size,
                                                                                  game_size,
                                                                                  snake_size
                                                                                  )
            best = population_evaluated[0]
            LOGGER.info(best)
            fraction = "{:.1f}_{:.1f}___{:.2f}".format(best.apples + snake_size, game_size ** 2,
                                                       (best.apples + snake_size) / game_size ** 2)
            file_name = "{}_____completion_{}_____movements_{:.1f}" \
                .format(i,
                        fraction,
                        best.movements)
            set_model_weights(self.model, best.model_genetic)
            data_utils.save_model(self.model, dir_path, file_name)
            population_genetic = new_population_genetic
            gc.collect()

    @timeit
    def execute_iteration(self, population_genetic, games_to_play_per_individual, selection_threshold, mutation_rate,
                          population_size,
                          game_size=6,
                          snake_size=4):
        game_statuses = [create_random_game_seed(game_size, snake_size) for j in range(games_to_play_per_individual)]
        # Evaluate population
        population_evaluated = evaluate_population(population_genetic, game_statuses, self.model, self.training_generator)
        # Select the best
        top_population_evaluated = elitism_selection(selection_threshold, population_evaluated)
        top_population_models = [top.model_genetic for top in top_population_evaluated]
        # Reproduce them
        number_of_children = population_size - len(top_population_evaluated)
        children = crossover(top_population_evaluated, number_of_children)
        # Introduce mutations
        mutate(children, mutation_rate)
        # The new generation contains best parents and children
        new_generation_models = children + top_population_models
        return new_generation_models, top_population_evaluated

    def build_model(self):
        tf.keras.backend.set_floatx('float64')
        if len(self.layers_size) < 3:
            raise ValueError("Incorrect number of layers")
        layer_lst = [layers.Dense(self.layers_size[1], activation='relu', input_shape=[self.layers_size[0]])]
        for i in range(2, len(self.layers_size)):
            layer_lst.append(layers.Dense(self.layers_size[i]))
        model = keras.Sequential(layer_lst)
        tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        return model

    def load_initial_population_genetic(self, model_paths: List[str], population_size,
                                        mutation_rate, games_to_play_per_individual,
                                        game_size):
        top_population_models = self.load_from_path(model_paths)
        # Reproduce them
        number_of_children = population_size - len(top_population_models)
        game_statuses = [create_random_game_seed(game_size, 4) for j in range(games_to_play_per_individual)]
        # Evaluate population
        population_evaluated = evaluate_population(top_population_models, game_statuses, self.model, self.training_generator)

        children = crossover(population_evaluated, number_of_children)
        # Introduce mutations
        mutate(children, mutation_rate)

        new_generation_models = children + top_population_models
        return new_generation_models

    def load_from_path(self, model_paths: List[str]):
        population_genetic = []
        for path in model_paths:
            model_genetic = []
            model = load_model(path)
            model_weights = model.get_weights()
            for j in range(len(model_weights)):
                model_genetic = model_weights
            population_genetic.append(model_genetic)
        return population_genetic
