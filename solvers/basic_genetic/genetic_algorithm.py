import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model.game_seed_creator import create_random_game_seed
from model.game_status import GameStatus
from solvers.basic_dnn import training_data_generator
from utils.timing import timeit


class GeneticAlgorithm:
    DATA_DIR = "C:\\Users\\Denis\\Desktop\\SnakePython\\data\\basic_genetic\\"

    def __init__(self, layers_size: List[int]):
        self.layers_size = layers_size
        self.number_of_layers = len(self.layers_size)
        self.model = self.build_model()

    def execute_iteration(self, population_genetic, games_to_play_per_individual, selection_threshold, mutation_rate):
        game_statuses = [create_random_game_seed(6, 2) for j in range(games_to_play_per_individual)]
        population_evaluated = self.evaluate_population(population_genetic, game_statuses)
        selected_pairs = self.selection(selection_threshold, population_evaluated)
        children = []
        even_masks = []
        odd_masks = []
        for layer in population_genetic[0]:
            even_masks.append(self._mask(layer.shape, True))
            odd_masks.append(self._mask(layer.shape, False))
        for pair in selected_pairs:
            children += self.couple_crossover(pair[0], pair[1], even_masks, odd_masks)
        for i in range(len(children)):
            self.mutation(mutation_rate, children[i])
        return children, population_evaluated

    def get_initial_population_genetic(self, population_size: int) -> List:
        """
        Each element in the population is a neural network
        with the dimensions defined in self.layers_size.
        For each layer we need to initialize the weights.
        The weights of each layer have dimensionality: [size of previous layer, size of current layer].
        Notice that we are not tuning the biases of each layer.
        :param population_size:
        :return: List of population size containing model_genetics. Each model_genetic is a list
        of weights for each layer in the neural network.
        """
        population_genetic = []
        for i in range(population_size):
            # For each layer we need to create the original parameters
            # We do not consider bias
            model_genetic = []
            for j in range(1, self.number_of_layers):
                rows = self.layers_size[j - 1]
                columns = self.layers_size[j]
                weights = np.random.uniform(low=-1, high=1, size=(rows, columns))
                model_genetic.append(weights)
            population_genetic.append(model_genetic)
        return population_genetic

    @timeit
    def evaluate_population(self, population_genetic: List, game__status_seeds: List[GameStatus]) -> List[list]:
        """
        :param population_genetic: List of population size containing model_genetics. Each model_genetic is a list
        of weights for each layer in the neural network.
        :param game_status_seeds: List of GameStatus that will be used for the initial state of the games that will
        be played by each model_genetic.
        :return: List sorted descending based on fitness. The list contains [[fitness, snake_length, movements, model_genetics], [...]]
        """
        population_evaluated = []
        # For each model genetic we need to perform artificial games and see the performance
        for model_genetic in population_genetic:
            self._set_model_weights(self.model, model_genetic)
            model_fitness, snake_length, number_of_movements = self.evaluate_model(game__status_seeds, self.model)
            population_evaluated.append([model_fitness, snake_length, number_of_movements, model_genetic])
        population_evaluated.sort(key=lambda x: x[0], reverse=True)
        best = population_evaluated[0]
        print(
            "     - Best fitness={} with snake length={} and number of movements={}".format(best[0], best[1], best[2]))
        return population_evaluated

    def _set_model_weights(self, model, model_genetic):
        weights = model.get_weights()
        # We do not consider bias
        for i in range(0, len(weights), 2):
            if weights[i].shape != model_genetic[i // 2].shape:
                raise ValueError("Error setting model shapes")
            weights[i] = model_genetic[i // 2]
        model.set_weights(weights)

    def evaluate_model(self, game_status_seeds: List[GameStatus], model) -> Tuple[int, int, int]:
        model_fitness: int = 0
        number_of_movements: int = 0
        snake_length: int = 0
        for game_status in game_status_seeds:
            game_statuses = []
            counter = 250
            while game_status.is_valid_game() and counter > 0:
                counter -= 1
                input = training_data_generator.get_input_from_game_status(game_status)
                dir = self.get_best_movement(input, model)
                new_game_status = game_status.move(dir)
                game_statuses.append(new_game_status)
                # Evaluate fitness
                reward = training_data_generator.get_reward(game_status, new_game_status)
                model_fitness += reward
                # Continue iteration
                game_status = new_game_status
            number_of_movements += len(game_statuses)
            snake_length += len(game_status.snake)
        snake_length = snake_length // len(game_status_seeds)
        number_of_movements = number_of_movements // len(game_status_seeds)
        return model_fitness, snake_length, number_of_movements

    def get_best_movement(self, input, model):
        test_predictions = model.predict(input).flatten()
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def selection(self, threshold: float, population_evaluated: List):
        """
        Taken as input the population evaluated with a specific fitness it outputs a list of couples for future
        reproductions based on the threshold and roulette selection.
        :param threshold: float between 0 and 1
        :param population_evaluated: The list contains [[fitness, model_genetics], [fitness_2, model_genetics_2], ...]
        :return: A list of pairs of the best evaluated models [[model_evaluated_father, model_evaluated_mother] , [...]]
        model_evaluted = [fitness, model_genetic]
        """
        population_size = len(population_evaluated)
        number_of_top_performers = int(threshold * len(population_evaluated))
        top_performers = population_evaluated[-number_of_top_performers:]
        pairs = []
        # Ensure we do not lose our best
        for top in top_performers:
            pairs.append([top[3], top[3]])
        min_fitness = min(map(lambda x: x[0], top_performers))
        total_fitness = sum([x[0] for x in top_performers])
        while len(pairs) < population_size // 2:
            pairs.append(self._pair([[row[0], row[3]] for row in top_performers], min_fitness, total_fitness))
        return pairs

    def couple_crossover(self, model_genetic_father, model_genetic_mother, masks_even, masks_odd):
        child_a = []
        child_b = []
        for layer_index in range(len(model_genetic_father)):
            layer_father = model_genetic_father[layer_index]
            layer_mother = model_genetic_mother[layer_index]
            layer_a = layer_father * masks_even[layer_index] + layer_mother * masks_odd[layer_index]
            layer_b = layer_mother * masks_even[layer_index] + layer_father * masks_odd[layer_index]
            child_a.append(layer_a)
            child_b.append(layer_b)
        return [child_a, child_b]

    def mutation(self, mutation_rate, model_genetic):
        for layer_index in range(len(model_genetic)):
            with np.nditer(model_genetic[layer_index], op_flags=['writeonly']) as it:
                for x in it:
                    if np.random.choice([True, False], p=[mutation_rate, 1 - mutation_rate]):
                        x[...] = random.uniform(-1, 1)
        return model_genetic

    def _pair(self, parents, min_fitness, total_fitness):
        """

        :param parents: List of all the model_genetics with fitness
        :param min_fitness: min fitness across population
        :param total_fitness: sum of all the fitness across the population
        :return: A random couple chosen based on roulete selection
        """
        pick = random.uniform(min_fitness, total_fitness)
        return [self._roulette_selection(parents, pick), self._roulette_selection(parents, pick)]

    def _roulette_selection(self, parents, pick):
        current = 0
        for parent in parents:
            current += abs(parent[0])
            if current > pick:
                return parent[1]

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(self.layers_size[1], activation='relu', input_shape=[self.layers_size[0]]),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        # model.summary()
        return model

    def _mask(self, shape: Tuple, mask_even: bool = True):
        """
        Example: shape(1, 4), mask_even=True -> [0, 1, 0, 1]
        :param shape: shape of the array as tuple.
        :param mask_even: if true the even positions of the the returned array will be 0 and the odd positions 1. False otherwise.
        :return: numpy array of the indicated shape with 1s and 0s.
        """
        if len(shape) != 2:
            raise ValueError("Only 2 dimensional matrix are supported: {}".format(shape))
        val = 0 if mask_even else 1
        return np.fromfunction(lambda i, j: (val + i + j) % 2, shape, dtype=int)

    def run(self, population_size, selection_threshold, mutation_rate, iterations, games_to_play_per_individual=1):
        model_description = "pop={}_sel={}_mut_{}_it_{}_games_{}\\".format(population_size, selection_threshold, mutation_rate, iterations, games_to_play_per_individual)
        dir_path = self.DATA_DIR + model_description
        os.mkdir(dir_path)
        population_genetic = self.get_initial_population_genetic(population_size)
        # Iterate
        for i in range(iterations):
            print("Iteration: {}".format(i))
            new_population_genetic, population_evaluated = self.execute_iteration(population_genetic, games_to_play_per_individual,
                                                                                  selection_threshold, mutation_rate)
            self.save_model(self.model, dir_path, population_evaluated, i)
            population_genetic = new_population_genetic

    @timeit
    def save_model(self, model, folder_path: str, population_evaluated, iteration):
        best = population_evaluated[0]
        file_name = "{}_iterations_fitness_{}_snake_lenght_{}_movements_{}".format(iteration, best[0], best[1], best[2])
        full_file_path = folder_path + file_name
        model.save(full_file_path)