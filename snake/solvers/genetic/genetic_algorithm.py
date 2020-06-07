import gc
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game.game_seed_creator import create_random_game_seed
from gui.gui_starter import get_models_from_path
from solvers.genetic.crossover import crossover
from solvers.genetic.evaluation import evaluate_population, set_model_weights
from solvers.genetic.hyperparameters import HyperParameters
from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated
from solvers.genetic.mutation import mutate
from solvers.genetic.selection import elitism_selection
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.data_utils import load_model
from utils import aws_snake_utils
from utils.snake_logger import get_module_logger
from utils.timing import timeit

LOGGER = get_module_logger(__name__)


def generate_report(previous_top_population: List[ModelGeneticEvaluated]):
    hyperparameters = None
    iteration = 0
    time = 0
    # Average and best
    completion = 0
    apples = 0
    fitness = 0
    movements = 0

    # Moving average of average and best for previous parameters

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

    def run(self, h: HyperParameters, checkpoint_path):
        LOGGER.info("Running game: {}".format(h))
        hyperparameters_description = "pop={}_sel={}_mut_{}_it_{}_game_size_{}/" \
            .format(h.population_size,
                    h.selection_threshold,
                    h.mutation_rate,
                    h.iterations,
                    h.game_size)
        dir_path = aws_snake_utils.get_training_output_folder() + hyperparameters_description
        LOGGER.info("Saving output on: " + dir_path)
        if checkpoint_path is None:
            population_genetic = self.get_random_initial_population_genetic(h.population_size)
        else:
            population_genetic = self.load_from_checkpoint(checkpoint_path, h)
        # Iterate
        for i in range(h.iterations):
            snake_size = h.snake_size
            LOGGER.info("Running iteration: {}".format(i))
            new_population_genetic, previous_top_population = self.execute_iteration(population_genetic, h)
            best = previous_top_population[0]
            report = generate_report(previous_top_population)
            LOGGER.info(best)
            fraction = "{:.1f}_{:.1f}___{:.2f}".format(best.apples + snake_size, h.game_size ** 2,
                                                       (best.apples + snake_size) / h.game_size ** 2)
            file_name = "{}_____completion_{}_____movements_{:.1f}" \
                .format(i,
                        fraction,
                        best.movements)
            set_model_weights(self.model, best.model_genetic)
            data_utils.save_model(self.model, dir_path, file_name)
            population_genetic = new_population_genetic
            gc.collect()

    @timeit
    def execute_iteration(self, population_genetic, h: HyperParameters):
        games_to_play_per_individual = h.games_to_play

        game_statuses = [create_random_game_seed(h.game_size, h.snake_size) for j in
                         range(games_to_play_per_individual)]
        # Evaluate population
        population_evaluated = evaluate_population(population_genetic, game_statuses, self.model,
                                                   self.training_generator)
        # Select the best
        top_population_evaluated = elitism_selection(h.selection_threshold, population_evaluated)
        top_population_models = [top.model_genetic for top in top_population_evaluated]
        # Reproduce them
        number_of_children = h.population_size - len(top_population_evaluated)
        children = crossover(top_population_evaluated, number_of_children, h)
        # Introduce mutations
        mutate(children, h)
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

    def load_from_checkpoint(self, checkpoint_path: str, h: HyperParameters):
        top_population_models = self.load_from_path(checkpoint_path)
        # Reproduce them
        number_of_children = h.population_size - len(top_population_models)
        game_statuses = [create_random_game_seed(h.game_size, 4) for j in range(h.games_to_play)]
        # Evaluate population
        population_evaluated = evaluate_population(top_population_models, game_statuses, self.model,
                                                   self.training_generator)
        # Crossover
        children = crossover(population_evaluated, number_of_children, h)
        # Introduce mutations
        mutate(children, h)
        # New generation
        new_generation_models = children + top_population_models
        return new_generation_models

    def load_from_path(self, checkpoint_path: str):
        model_paths = get_models_from_path(checkpoint_path)
        population_genetic = []
        for path in model_paths:
            model_genetic = []
            model = load_model(path)
            model_weights = model.get_weights()
            for j in range(len(model_weights)):
                model_genetic = model_weights
            population_genetic.append(model_genetic)
        return population_genetic
