import unittest
import numpy as np

from solvers.basic_genetic_evolution.genetic_algorithm import GeneticAlgorithm


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
        self.assertEqual(1, len(population_genetic))
        self.assertEqual((9, 125), layer_0_genetic_size)
        self.assertEqual((125, 1), layer_1_genetic_size)

    def test_generate_initial_population_genetic_values(self):
        # Arrange
        population_size = 1
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

    def test_evaluate_performance(self):
        # Arrange
        population_size = 2
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        # Act
        # population_with_fitness = self.ga.evaluate_population(population_genetic)
        # Assert

