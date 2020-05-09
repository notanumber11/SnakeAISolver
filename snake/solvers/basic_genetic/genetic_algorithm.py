import random
from typing import List, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from game.game_seed_creator import create_random_game_seed
from game.game_status import GameStatus
from solvers.training import basic_training_data_generator, training_utils
from solvers.basic_genetic.model_genetic_evaluated import ModelGeneticEvaluated
from utils import aws_snake_utils
from utils.snake_logger import get_module_logger
from utils.timing import timeit

LOGGER = get_module_logger(__name__)


class GeneticAlgorithm:

    def __init__(self, layers_size: List[int]):
        self.layers_size = layers_size
        self.number_of_layers = len(self.layers_size)
        self.model = self.build_model()

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
            # The first layer is the input so we ignore it
            model_genetic = []
            for j in range(1, self.number_of_layers):
                rows = self.layers_size[j - 1]
                columns = self.layers_size[j]
                weights = np.random.uniform(low=-1, high=1, size=(rows, columns))
                model_genetic.append(weights)
            population_genetic.append(model_genetic)
        return population_genetic

    @timeit
    def evaluate_population(self, population_genetic: List, game_status_seeds: List[GameStatus]) \
            -> List[ModelGeneticEvaluated]:
        """
        :param population_genetic: List of population size containing model_genetics. Each model_genetic is a list
        of weights for each layer in the neural network.
        :param game_status_seeds: List of GameStatus that will be used for the initial state of the games that will
        be played by each model_genetic.
        :return: List sorted descending based on fitness.
        """
        population_evaluated = []
        # For each game genetic we need to perform artificial games and see the performance
        for model_genetic in population_genetic:
            model_evaluated = self.evaluate_model(game_status_seeds, self.model, model_genetic)
            population_evaluated.append(model_evaluated)
        return population_evaluated

    def _set_model_weights(self, model, model_genetic):
        weights = model.get_weights()
        # We do not consider bias
        for i in range(0, len(model_genetic)):
            if weights[i * 2].shape != model_genetic[i].shape:
                raise ValueError("Error setting game shapes")
            weights[i * 2] = model_genetic[i]
        model.set_weights(weights)

    def evaluate_model(self, game_status_seeds: List[GameStatus], model, model_genetic) -> ModelGeneticEvaluated:
        self._set_model_weights(self.model, model_genetic)
        reward = 0
        number_of_movements = 0
        snake_length = 0
        for game_status in game_status_seeds:
            # Play one game
            game_snake_length, game_movements, game_reward = self.play_one_game(game_status, model)
            # Finish playing one game
            number_of_movements += game_movements
            snake_length += game_snake_length
            reward += game_reward

        snake_length /= len(game_status_seeds)
        number_of_movements /= len(game_status_seeds)
        reward /= len(game_status_seeds)
        return ModelGeneticEvaluated(snake_length, number_of_movements, reward, game_status_seeds[0].size,
                                     model_genetic)

    def play_one_game(self, game_status_seed: GameStatus, model):
        game_statuses = []
        movements_left = game_status_seed.get_number_of_holes()
        accumulated_reward = 0
        while game_status_seed.is_valid_game() and movements_left > 0:
            _input = basic_training_data_generator.get_input_from_game_status(game_status_seed)
            _dir = self.get_best_movement(_input, model)
            new_game_status = game_status_seed.move(_dir)
            game_statuses.append(new_game_status)
            # Evaluate fitness
            reward = basic_training_data_generator.get_reward(game_status_seed, new_game_status)
            if reward == 0.7:
                movements_left = game_status_seed.get_number_of_holes()
            accumulated_reward += reward
            # Continue iteration
            game_status_seed = new_game_status
            movements_left -= 1
        snake_length = len(game_statuses[-1].snake)
        if movements_left == 0:
            snake_length = 0
        return snake_length, len(game_statuses), accumulated_reward

    def get_best_movement(self, input, model):
        test_predictions = model.predict(np.array(input), batch_size=4).flatten()
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def selection(self, threshold: float, population_evaluated: List[ModelGeneticEvaluated]) -> List[List]:
        """
        Taken as input the population evaluated with a specific fitness it outputs a list of couples for future
        reproductions based on the threshold and roulette selection.
        :param threshold: float between 0 and 1
        :param population_evaluated: The list contains [[fitness, model_genetics], [fitness_2, model_genetics_2], ...]
        :return: A list of pairs of the best evaluated models [[model_evaluated_father, model_evaluated_mother] , [...]]
        model_evaluted = [fitness, model_genetic]
        """
        population_evaluated.sort(key=lambda x: x.fitness(), reverse=True)
        population_size = len(population_evaluated)
        number_of_top_performers = int(threshold * len(population_evaluated))
        top_performers = population_evaluated[0:number_of_top_performers]
        pairs = []
        # Ensure we do not lose our best
        # for top in top_performers:
        #     pairs.append([top[3], top[3]])
        total_fitness = sum([x.fitness() for x in top_performers])
        while len(pairs) < population_size // 2:
            pair = self._pair(top_performers, total_fitness)
            pairs.append(pair)
        return pairs

    def _pair(self, parents: List[ModelGeneticEvaluated], total_fitness: float) -> List[ModelGeneticEvaluated]:
        """

        :param parents: List of all the model_genetics with fitness
        :param min_fitness: min fitness across population
        :param total_fitness: sum of all the fitness across the population
        :return: A random couple chosen based on roulete selection
        """
        pick_1 = random.uniform(0, total_fitness)
        pick_2 = random.uniform(0, total_fitness)
        return [self._roulette_selection(parents, pick_1), self._roulette_selection(parents, pick_2)]

    def _roulette_selection(self, parents: List[ModelGeneticEvaluated], pick: float) -> ModelGeneticEvaluated:
        current = 0
        for parent in parents:
            current += parent.fitness()
            if current >= pick:
                return parent
        raise ValueError("Error performing roulette selection with pick={} and parents={}".format(pick, parents))

    def crossover(self, selected_pairs: List[List[ModelGeneticEvaluated]], fix_crossover=False):
        children = []
        if fix_crossover:
            even_masks = []
            odd_masks = []
            for layer in selected_pairs[0][0].model_genetic:
                even_masks.append(self._mask(layer.shape, True))
                odd_masks.append(self._mask(layer.shape, False))
            for pair in selected_pairs:
                children += self._fix_crossover(pair[0].model_genetic, pair[1].model_genetic, even_masks, odd_masks)
            return children
        else:
            for pair in selected_pairs:
                children += self._random_crossover(pair[0].model_genetic, pair[1].model_genetic)
            return children

    def _fix_crossover(self, model_genetic_father, model_genetic_mother, masks_even, masks_odd):
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

    def _random_crossover(self, model_genetic_father, model_genetic_mother):
        child_a = []
        child_b = []
        for layer_index in range(len(model_genetic_father)):
            layer_father = model_genetic_father[layer_index]
            layer_mother = model_genetic_mother[layer_index]
            mask_a = np.random.choice([0, 1], size=layer_father.shape)
            mask_b = np.where(mask_a < 1, 1, 0)
            layer_a = layer_father * mask_a[layer_index] + layer_mother * mask_b[layer_index]
            layer_b = layer_mother * mask_a[layer_index] + layer_father * mask_b[layer_index]
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

    def _mask(self, shape: Tuple, mask_even: bool = True):
        """
        Example: shape(1, 4), mask_even=True -> [0, 1, 0, 1]
        :param shape: shape of the array as tuple.
        :param mask_even: if true the even positions of the the returned array will be 0 and the odd positions 1. False otherwise.
        :return: numpy array of the indicated shape with 1s and 0s.
        """
        if len(shape) != 2:
            raise ValueError("Only 2 dimensional matrix are supported: {}".format(shape))
        width = shape[1]
        return np.fromfunction(lambda i, j: (i * width + j) % 2 == mask_even, shape, dtype=int)

    def run(self, population_size, selection_threshold, mutation_rate, iterations, games_to_play_per_individual=1,
            game_size=6):
        model_description = "pop={}_sel={}_mut_{}_it_{}_games_{}\\".format(population_size, selection_threshold,
                                                                           mutation_rate, iterations,
                                                                           games_to_play_per_individual)
        LOGGER.info("Running game: {}".format(model_description))
        dir_path = aws_snake_utils.get_training_output_folder() + model_description
        population_genetic = self.get_initial_population_genetic(population_size)
        # Iterate
        for i in range(iterations):

            LOGGER.info("Running iteration: {}".format(i))
            new_population_genetic, population_evaluated = self.execute_iteration(population_genetic,
                                                                                  games_to_play_per_individual,
                                                                                  selection_threshold, mutation_rate,
                                                                                  game_size)
            best = population_evaluated[0]
            LOGGER.info(best)
            file_name = "{:.2f}_iterations_fitness_{:.2f}_snake_length_{:.2f}_movements_{:.2f}".format(i,
                                                                                       best.fitness(),
                                                                                       best.snake_length,
                                                                                       best.movements)
            self._set_model_weights(self.model, best.model_genetic)
            training_utils.save_model(self.model, dir_path, file_name)
            population_genetic = new_population_genetic

    @timeit
    def execute_iteration(self, population_genetic, games_to_play_per_individual, selection_threshold, mutation_rate,
                          game_size=6):
        game_statuses = [create_random_game_seed(game_size, 2) for j in range(games_to_play_per_individual)]
        # Evaluate population
        population_evaluated = self.evaluate_population(population_genetic, game_statuses)
        # Select best couples
        selected_pairs = self.selection(selection_threshold, population_evaluated)
        # Reproduce them
        children = self.crossover(selected_pairs)
        # Introduce mutations
        for i in range(len(children)):
            self.mutation(mutation_rate, children[i])
        return children, population_evaluated

    def build_model(self):
        if len(self.layers_size) < 3:
            raise ValueError("Incorrect number of layers")
        layer_lst = [layers.Dense(self.layers_size[1], activation='relu', input_shape=[self.layers_size[0]])]
        for i in range(2, len(self.layers_size)):
            layer_lst.append(layers.Dense(self.layers_size[i]))

        model = keras.Sequential(layer_lst)

        # optimizer = tf.keras.optimizers.RMSprop(0.001)
        # game.compile(loss='mse',
        #               optimizer=optimizer,
        #               metrics=['mae', 'mse'])
        # game.summary()
        return model
