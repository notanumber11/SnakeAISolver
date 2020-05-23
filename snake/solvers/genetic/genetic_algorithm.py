import random
from typing import List

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from game.game import Game
from game.game_seed_creator import create_random_game_seed
from game.game_status import GameStatus
from solvers.genetic.advance_genetic_model_evaluated import AdvanceModelGeneticEvaluated
from solvers.genetic.crossover import random_crossover, simulated_binary_crossover, single_point_binary_crossover
from solvers.genetic.mutation import uniform_mutation, gaussian_mutation
from solvers.genetic.basic_genetic_solver import BasicGeneticSolver
from solvers.training import basic_training_data_generator as training_generator, training_utils
from utils import aws_snake_utils
from utils.snake_logger import get_module_logger
from utils.timing import timeit
import tensorflow as tf
LOGGER = get_module_logger(__name__)


class GeneticAlgorithm:

    def __init__(self, layers_size: List[int]):
        self.layers_size = layers_size
        self.number_of_layers = len(self.layers_size)
        self.model = self.build_model()
        self.training_generator = training_generator

    def get_initial_population_genetic(self, population_size: int) -> List:
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

    def _set_model_weights(self, model, model_genetic):
        weights = model.get_weights()
        for i in range(0, len(model_genetic)):
            weights[i] = model_genetic[i]
        model.set_weights(weights)

    def evaluate_population(self, population_genetic: List, game_status_seeds: List[GameStatus]) \
            -> List[AdvanceModelGeneticEvaluated]:
        population_evaluated = []
        # For each game genetic we need to perform artificial games and see the performance
        for model_genetic in population_genetic:
            model_evaluated = self.evaluate_model(game_status_seeds, self.model, model_genetic)
            population_evaluated.append(model_evaluated)
        return population_evaluated

    def evaluate_model(self, game_status_seeds: List[GameStatus], model, model_genetic) -> AdvanceModelGeneticEvaluated:
        self._set_model_weights(self.model, model_genetic)
        games = []
        for game_status in game_status_seeds:
            game = self.play_one_game(game_status, model, self.training_generator)
            games.append(game)
        return AdvanceModelGeneticEvaluated(games, model_genetic)

    def play_one_game(self, current_game_status: GameStatus, model, training_generator):
        game_statuses = [current_game_status]
        movements_left = current_game_status.get_number_of_holes()
        while current_game_status.is_valid_game() and movements_left > 0:
            _input = [training_generator.get_input_from_game_status(current_game_status)]
            _dir = self.get_best_movement(_input, model)
            new_game_status = current_game_status.move(_dir)
            game_statuses.append(new_game_status)
            # Continue iteration
            movements_left -= 1
            if current_game_status.apple != new_game_status.apple:
                movements_left = new_game_status.get_number_of_holes()
            current_game_status = new_game_status

        # The game was in a loop
        is_loop = True if movements_left == 0 and current_game_status.is_full_finished() is False else False
        game = Game(game_statuses, is_loop)
        return game

    def get_best_movement(self, input, model):
        test_predictions = model.__call__(np.array(input[0]))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def elitism_selection(self, percentage: float, population_evaluated: List[AdvanceModelGeneticEvaluated]) -> List[AdvanceModelGeneticEvaluated]:
        population_evaluated.sort(key=lambda x: x.fitness(), reverse=True)
        number_of_top_performers = int(percentage * len(population_evaluated))
        top_performers = population_evaluated[0:number_of_top_performers]
        return top_performers

    def _pair(self, parents: List[AdvanceModelGeneticEvaluated], total_fitness: float) -> List[AdvanceModelGeneticEvaluated]:
        """

        :param parents: List of all the model_genetics with fitness
        :param min_fitness: min fitness across population
        :param total_fitness: sum of all the fitness across the population
        :return: A random couple chosen based on roulete selection
        """
        pick_1 = random.uniform(0, total_fitness)
        pick_2 = random.uniform(0, total_fitness)
        return [self._roulette_selection(parents, pick_1), self._roulette_selection(parents, pick_2)]

    def _roulette_selection(self, parents: List[AdvanceModelGeneticEvaluated], pick: float) -> AdvanceModelGeneticEvaluated:
        current = 0
        for parent in parents:
            current += parent.fitness()
            if current >= pick:
                return parent
        raise ValueError("Error performing roulette selection with pick={} and parents={}".format(pick, parents))

    def crossover(self, top_performers: List[AdvanceModelGeneticEvaluated], number_of_children):
        total_fitness = sum([x.fitness() for x in top_performers])
        children = []
        cross_type = {
            "random": 0.25,
            "single_point_binary": 0.25,
            "simulated_binary": 0.5
        }

        cross_functions = {
            "random": random_crossover,
            "single_point_binary": single_point_binary_crossover,
            "simulated_binary": simulated_binary_crossover
        }
        options = list(cross_type.keys())
        probabilities = list(cross_type.values())
        while len(children) <= number_of_children:
            pair = self._pair(top_performers, total_fitness)
            choice = np.random.choice(options, p=probabilities)
            children += cross_functions[choice](pair[0].model_genetic, pair[1].model_genetic)

        return children[:number_of_children]

    def run(self, population_size, selection_threshold, mutation_rate, iterations, games_to_play_per_individual=1,
            game_size=6):
        model_description = "pop={}_sel={}_mut_{}_it_{}_games_{}_game_size_{}/".format(population_size,
                                                                                       selection_threshold,
                                                                                       mutation_rate, iterations,
                                                                                       games_to_play_per_individual,
                                                                                       game_size)
        LOGGER.info("Running game: {}".format(model_description))
        dir_path = aws_snake_utils.get_training_output_folder() + model_description
        population_genetic = self.get_initial_population_genetic(population_size)
        # Iterate
        for i in range(iterations):
            LOGGER.info("Running iteration: {}".format(i))
            new_population_genetic, population_evaluated = self.execute_iteration(population_genetic,
                                                                                  games_to_play_per_individual,
                                                                                  selection_threshold, mutation_rate,
                                                                                  population_size,
                                                                                  game_size
                                                                                  )
            best = population_evaluated[0]
            LOGGER.info(best)
            file_name = "{:.2f}_iterations_fitness_{:.2f}_snake_length_{:.2f}_movements_{:.2f}" \
                .format(i,
                        best.fitness(),
                        best.snake_length,
                        best.movements)
            self._set_model_weights(self.model, best.model_genetic)
            path = training_utils.save_model(self.model, dir_path, file_name)
            self.show_current_best_model(i, path, game_size)

            population_genetic = new_population_genetic


    def show_current_best_model(self, iteration, path, game_size):
        if aws_snake_utils.is_local_run() and iteration % 25 == 0:
            from gui.gui_starter import show_solver
            show_solver(BasicGeneticSolver(path), game_size, 3, 6)

    @timeit
    def execute_iteration(self, population_genetic, games_to_play_per_individual, selection_threshold, mutation_rate,
                          population_size,
                          game_size=6):
        game_statuses = [create_random_game_seed(game_size, 4) for j in range(games_to_play_per_individual)]
        # Evaluate population
        population_evaluated = self.evaluate_population(population_genetic, game_statuses)
        # Select the best
        top_population = self.elitism_selection(selection_threshold, population_evaluated)
        top_population_models = [top.model_genetic for top in top_population]
        # Reproduce them
        number_of_children = population_size - len(top_population)
        children = self.crossover(top_population, number_of_children)
        # Introduce mutations
        self.mutate(children, mutation_rate)

        new_generation_models = children + top_population_models
        return new_generation_models, top_population

    def mutate(self, children, mutation_rate):
        mut_type = {
            "uniform": 0.33,
            "gaussian": 0.67
        }
        options = list(mut_type.keys())
        probabilities = list(mut_type.values())
        for i in range(len(children)):
            choice = np.random.choice(options, p=probabilities)
            if choice == "uniform":
                uniform_mutation(children[i], mutation_rate)
            elif choice == "gaussian":
                gaussian_mutation(children[i], mutation_rate)
            else:
                raise ValueError("Incorrect choice: " + choice)

    def build_model(self):
        tf.keras.backend.set_floatx('float64')
        if len(self.layers_size) < 3:
            raise ValueError("Incorrect number of layers")
        layer_lst = [layers.Dense(self.layers_size[1], activation='relu', input_shape=[self.layers_size[0]])]
        for i in range(2, len(self.layers_size)):
            layer_lst.append(layers.Dense(self.layers_size[i]))
        model = keras.Sequential(layer_lst)
        return model
