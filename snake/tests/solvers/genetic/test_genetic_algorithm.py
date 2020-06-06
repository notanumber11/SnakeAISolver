import unittest
from unittest.mock import MagicMock

import numpy
import numpy as np
from numpy.testing import assert_raises

from game import game_seed_creator
from game.game_seed_creator import create_default_game_seed, create_random_game_seed
from solvers.training_data_generators.classification.binary_vision import BinaryVision
from solvers.genetic.crossover import pair
from solvers.genetic.evaluation import evaluate_population, set_model_weights, play_one_game, \
    get_best_movement
from solvers.genetic.genetic_algorithm import GeneticAlgorithm
from solvers.genetic.mutation import mutate
from solvers.genetic.selection import elitism_selection


class GeneticAlgorithmTest(unittest.TestCase):

    def setUp(self):
        #self.layers_size = [9, 125, 1]
        self.layers_size = [28, 20, 12, 4]
        self.ga = GeneticAlgorithm(self.layers_size, BinaryVision())

    def test_generate_initial_population_layer_sizes(self):
        # Arrange
        population_size = 2
        # Act
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        model_genetic = population_genetic[0]
        layer_0_genetic_size = model_genetic[0].shape
        # Assert
        self.assertEqual(population_size, len(population_genetic))
        self.assertEqual(len(population_genetic[0]) / 2, len(self.layers_size) - 1)
        self.assertEqual((self.layers_size[0], self.layers_size[1]), layer_0_genetic_size)

    def test_generate_initial_population_genetic_values(self):
        # Arrange
        population_size = 2
        # Act
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        # Assert
        model_genetic = population_genetic[0]
        all_values_flatten = [val.flatten() for val in model_genetic]
        all_values = np.concatenate(all_values_flatten)
        min = np.amin(all_values)
        max = np.amin(all_values)
        self.assertGreaterEqual(max, -1)
        self.assertLessEqual(min, 1)
        # Test that models generated randomly are different
        assert_raises(AssertionError, numpy.testing.assert_array_equal, population_genetic[0][0],
                      population_genetic[1][0])
        # All values are between -1 and 1
        for model_genetic in population_genetic:
            self.assertTrue(((-1 <= model_genetic[0]) & (model_genetic[0] <= 1)).all())

    def test_evaluate_population(self):
        # Arrange
        population_size = 3
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        # Act
        population_with_fitness = evaluate_population(population_genetic, game_statuses, self.ga.model, self.ga.training_generator)
        # Assert
        self.assertEqual(len(population_with_fitness), population_size)

    def test_evaluate_best_movement_with_different_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
        for i in range(100):
            model_genetic = self.ga.get_random_initial_population_genetic(1)[0]
            set_model_weights(self.ga.model, model_genetic)
            predictions.append(get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the weights are different
        self.assertFalse(all(x == predictions[0] for x in predictions))

    def test_evaluate_best_movement_with_same_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
        model_genetic = self.ga.get_random_initial_population_genetic(1)[0]
        set_model_weights(self.ga.model, model_genetic)
        for i in range(100):
            predictions.append(get_best_movement(_input, self.ga.model))
        # All predictions are the same since the weights are the same
        self.assertFalse(any(x != predictions[0] for x in predictions))

    def test_evaluate_best_movement_with_same_model_different_input(self):
        predictions = []
        model_genetic = self.ga.get_random_initial_population_genetic(1)[0]
        set_model_weights(self.ga.model, model_genetic)
        for i in range(100):
            game_status = create_random_game_seed(6, 2)
            _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
            predictions.append(get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the input is different
        self.assertFalse(all(x == predictions[0] for x in predictions))

    def test_selection_sort_population(self):
        # Arrange
        population_size = 10
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        population_with_fitness = evaluate_population(population_genetic, game_statuses, self.ga.model, self.ga.training_generator)
        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x.fitness(), reverse=True)
        threshold = 0.2
        # Act
        self.assertNotEqual(population_with_fitness, sorted_population_with_fitness)
        elitism_selection(threshold, population_with_fitness)
        self.assertEqual(population_with_fitness, sorted_population_with_fitness)

    def test_elitism_selection(self):
        # Arrange
        population_size = 10
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        population_with_fitness = evaluate_population(population_genetic, game_statuses, self.ga.model, self.ga.training_generator)
        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x.fitness(), reverse=True)
        threshold = 0.4
        number_of_top_performers = int(threshold * population_size)
        top_performers = sorted_population_with_fitness[0:number_of_top_performers]
        worst_performers = sorted_population_with_fitness[number_of_top_performers:]

        # Act
        selected_individuals = elitism_selection(threshold, population_with_fitness)

        # Assert
        self.assertEqual(len(selected_individuals), population_size * threshold)
        # All the worst performers are not included in the selected pairs
        self.assertTrue(all(
            population_with_fitness[number_of_top_performers].fitness() <= individual.fitness() for individual in
            selected_individuals))
        for individual in selected_individuals:
            self.assertFalse(any(individual == worst for worst in worst_performers),
                             msg="Error: The worst performers are present after selection phase")
            self.assertTrue(any(individual == top for top in top_performers),
                            msg="Error: No top performer is present after selection phase")
            return
        self.assertTrue(False)

    def test_pair_with_an_incredible_performer(self):
        # Arrange
        population_size = 20
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        population_with_fitness = evaluate_population(population_genetic, game_statuses, self.ga.model, self.ga.training_generator)
        for i in range(population_size):
            population_with_fitness[i].fitness = MagicMock(return_value=1)

        # One performer has significantly more fitness that the others
        population_with_fitness[-1].fitness = MagicMock(return_value=10000)
        total_fitness = sum([x.fitness() for x in population_with_fitness])

        top = population_with_fitness[-1]
        worst = population_with_fitness[0]
        boy, girl = pair(population_with_fitness, total_fitness)
        self.assertEqual(boy, girl)
        self.assertEqual(boy, top)
        self.assertNotEqual(boy, worst)
        self.assertTrue(all(boy != el for el in population_with_fitness[:-1]))

    def test_pair_equally(self):
        # Arrange
        population_size = 10
        fitness = 10
        total_fitness = 0
        number_of_pairs_to_generate = 50
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        population_with_fitness = evaluate_population(population_genetic, game_statuses, self.ga.model, self.ga.training_generator)

        # Each individual has 10% chances of being chosen since all of them share the same fitness
        for i in range(population_size):
            population_with_fitness[i].fitness = MagicMock(return_value=fitness)
            total_fitness += population_with_fitness[i].fitness()

        # If we obtain 100 elements the probability of one individual to be selected is ps = 1-(1/10)^100 ~= 1
        result = []
        for i in range(number_of_pairs_to_generate):
            parents = pair(population_with_fitness, total_fitness)
            result += parents

        self.assertEqual(number_of_pairs_to_generate * 2, len(result))
        self.assertTrue(all(el in result for el in population_with_fitness))

    def test_execute_iteration_with_single_element(self):
        games_to_play = 2
        population_size = 6
        # Create a population genetic with the same element repeated
        population_genetic = self.ga.get_random_initial_population_genetic(1) * population_size
        selection_threshold = 0.5
        # Do not mutate
        mutation_rate = 0
        new_generation, top_population = self.ga.execute_iteration(population_genetic, games_to_play,
                                                                   selection_threshold,
                                                                   mutation_rate, population_size)
        self.assertEqual(len(new_generation), population_size)

    def test_execute_iteration(self):
        games_to_play = 1
        population_size = 10
        # Create a population genetic with the same element
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)
        selection_threshold = 0.2
        mutation_rate = 1
        children, evaluation_result = self.ga.execute_iteration(population_genetic, games_to_play, selection_threshold,
                                                                mutation_rate, population_size)

    def test_mutate(self):
        population_genetic = [[np.ones(10)] for i in range(10)]
        mutate(population_genetic, 1)
        for model_genetic in population_genetic:
            for chromosome in model_genetic:
                equal = np.array_equal(chromosome, np.ones(10))
                self.assertFalse(equal, msg="Some chromosomes did not change")
        population_genetic = [[np.ones(10)] for i in range(10)]
        mutate(population_genetic, 0)
        for model_genetic in population_genetic:
            for chromosome in model_genetic:
                equal = np.array_equal(chromosome, np.ones(10))
                self.assertTrue(equal, msg="Chromosomes change with 0 mutation rate")

    def test_with_plus_parents(self):
        games_to_play = 1
        population_size = 10
        selection_threshold = 0.1
        mutation_rate = 0.05
        # Create a population genetic with the same element
        population_genetic = self.ga.get_random_initial_population_genetic(population_size)

        new_generation_models, top_population = self.ga.execute_iteration(population_genetic, games_to_play,
                                                                          selection_threshold,
                                                                          mutation_rate, population_size)
        top_population_models = [modelEvaluated.model_genetic for modelEvaluated in top_population]

        for top in top_population_models:
            self.assertTrue(any(self.equal_model_genetic(top, new_model) for new_model in new_generation_models))

    def equal_model_genetic(self, a, b):
        return all(np.array_equal(a[i], b[i]) for i in range(len(a)))

    def test_max_number_of_movements(self):
        game_status = game_seed_creator.create_default_game_seed()
        game_status.is_valid_game = MagicMock(return_value=True)
        game_status.move = MagicMock(return_value=game_status)
        game = play_one_game(game_status, self.ga.model, self.ga.training_generator)
        self.assertTrue(game.was_stack_in_loop)
        self.assertEqual(len(game.game_statuses), game_status.size ** 2 + len(game_status.snake) + 1)

    def test_absolute_distances(self):
        pass

    def direction_as_input(self):
        pass

    def use_hyperparemeters_as_setting(self):
        pass

    def checkout_points(self):
        pass

    def hyperparameter_search_hyperparameters(self):
        pass

    def make_basic_genetic_algorithm_classification(self):
        # this will allow to remove having two different classes for advance and basic genetic
        pass