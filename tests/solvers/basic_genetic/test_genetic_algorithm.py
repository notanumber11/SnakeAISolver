import math
import unittest

import numpy
import numpy as np
from numpy.testing import assert_raises

from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm
from model.game_seed_creator import create_default_game_seed
from utils.timing import timeit


class GeneticAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.layers_size = [9, 125, 1]
        self.ga = GeneticAlgorithm(self.layers_size)

    def test_generate_initial_population_layer_sizes(self):
        # Arrange
        population_size = 2
        # Act
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        model_genetic = population_genetic[0]
        layer_0_genetic_size = model_genetic[0].shape
        layer_1_genetic_size = model_genetic[1].shape
        # Assert
        self.assertEqual(2, len(population_genetic))
        self.assertEqual((9, 125), layer_0_genetic_size)
        self.assertEqual((125, 1), layer_1_genetic_size)

    def test_generate_initial_population_genetic_values(self):
        # Arrange
        population_size = 2
        # Act
        population_genetic = self.ga.get_initial_population_genetic(population_size)
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

    def test_evaluate_population(self):
        # Arrange
        population_size = 3
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        # Act
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)
        # Assert
        self.assertEqual(len(population_with_fitness), population_size)
        # Population is sorted by fitness
        for i in range(0, population_size - 1):
            self.assertGreaterEqual(population_with_fitness[i][0], population_with_fitness[i + 1][0])

    def test_selection(self):
        # Arrange
        population_size = 6
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)
        threshold = 0.5
        first_excluded_model = population_with_fitness[int(len(population_with_fitness) * threshold) + 1]
        # Act
        selected_pairs = self.ga.selection(threshold, population_with_fitness)
        # Assert
        self.assertEqual(len(selected_pairs), math.ceil(population_size / 2))

    def test_couple_crossover_same_parent(self):
        # Arrange
        population_size = 1
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        parent_model_genetic = population_genetic[0]
        mother_model_genetic = population_genetic[0]
        even_masks = []
        odd_masks = []
        for layer in parent_model_genetic:
            even_masks.append(self.ga._mask(layer.shape, True))
            odd_masks.append(self.ga._mask(layer.shape, False))

        # Act
        children = self.ga.couple_crossover(parent_model_genetic, mother_model_genetic, even_masks, odd_masks)

        # Assert
        for layer_index in range(len(parent_model_genetic)):
            np.testing.assert_array_equal(parent_model_genetic[layer_index], mother_model_genetic[layer_index])
            np.testing.assert_array_equal(parent_model_genetic[layer_index], children[0][layer_index])

    def test_couple_crossover_different_parent(self):
        # Arrange
        population_size = 2
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        parent_model_genetic = population_genetic[0]
        mother_model_genetic = population_genetic[1]
        even_masks = []
        odd_masks = []
        for layer in parent_model_genetic:
            even_masks.append(self.ga._mask(layer.shape, True))
            odd_masks.append(self.ga._mask(layer.shape, False))

        # Act
        children = self.ga.couple_crossover(parent_model_genetic, mother_model_genetic, even_masks, odd_masks)

        # Assert
        child_a = children[0]
        child_b = children[1]
        for layer_index in range(len(parent_model_genetic)):
            flat_parent_layer = parent_model_genetic[layer_index].flat
            flat_mother_layer = mother_model_genetic[layer_index].flat
            flat_a_child = child_a[layer_index].flat
            flat_b_child = child_b[layer_index].flat
            for i in range(len(flat_parent_layer)):
                if i % 2 == 1:
                    self.assertEqual(flat_parent_layer[i], flat_a_child[i])
                    self.assertEqual(flat_mother_layer[i], flat_b_child[i])
                else:
                    self.assertEqual(flat_parent_layer[i], flat_b_child[i])
                    self.assertEqual(flat_mother_layer[i], flat_a_child[i])

    def test_mask(self):
        mask_even = np.array([[0, 1, 0], [1, 0, 1]])
        mask_odd = np.array([[1, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(mask_even, self.ga._mask(mask_even.shape, True))
        np.testing.assert_array_equal(mask_odd, self.ga._mask(mask_odd.shape, False))

    def test_mutation(self):
        size = 1000
        mutation_rate = 0.05
        model_genetic = [np.ones(size)]
        # mutate 0
        self.ga.mutation(0, model_genetic)
        for i in range(len(model_genetic)):
            np.testing.assert_array_equal(model_genetic[i], np.ones(size))
        # mutate all
        self.ga.mutation(0.05, model_genetic)
        equal_count = 0
        for i in range(len(model_genetic)):
            for j in range(size):
                if model_genetic[i][j] == 1:
                    equal_count += 1
        self.assertGreaterEqual(equal_count, 5 * mutation_rate * size)
        self.assertLessEqual(equal_count, size * (1 - mutation_rate / 5))

    def test_execute_iteration(self):
        games_to_play = 5
        population_size = 6
        population_genetic = self.ga.get_initial_population_genetic(1) * population_size
        # Create a population genetic with the same elements
        selection_threshold = 0.5
        # Do not mutate
        mutation_rate = 0
        children, evaluation_result = self.ga.execute_iteration(population_genetic, games_to_play, selection_threshold,
                                                                mutation_rate)
        self.assertEqual(len(children), population_size)
        for child in children:
            for layer_index in range(len(child)):
                np.testing.assert_array_equal(child[layer_index], population_genetic[0][layer_index])

    def test_new_weights_modify_predictions(self):
        population_size = 5
        game_seeds_size = 1
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        game_seeds = [create_default_game_seed() for i in range(game_seeds_size)]
        models_evaluated = []
        for model_genetic in population_genetic:
            for layer_index in range(len(model_genetic)):
                # model_genetic[layer_index] = np.zeros(model_genetic[layer_index].shape)
                model_genetic[layer_index] = np.random.uniform(low=-1, high=1, size=model_genetic[layer_index].shape)
            self.ga._set_model_weights(self.ga.model, model_genetic)
            r = self.ga.evaluate_model(game_seeds, self.ga.model)
            models_evaluated.append(r)
        self.assertFalse(all(x == models_evaluated[0] for x in models_evaluated))
