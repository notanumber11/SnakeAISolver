import unittest
from unittest.mock import MagicMock

import numpy as np

from solvers.genetic.crossover import simulated_binary_crossover, single_point_binary_crossover, \
    random_crossover


class TestCrossover(unittest.TestCase):

    def setUp(self):
        self.layers_size = [28, 20, 12, 4]
        self.backup_np_random_random = np.random.random
        self.backup_random_random_int = np.random.randint

    def tearDown(self):
        np.random.random = self.backup_np_random_random
        np.random.randint = self.backup_random_random_int

    def test_simulated_binary_crossover(self):
        eta = 1000
        father = [
            np.array([-1 for i in range(10)])
        ]
        mother = [
            np.array([1 for i in range(10)])
        ]
        # When random value u is always 0.5 the children are exactly as the parents
        np.random.random = MagicMock(return_value=np.full(father[0].shape, 0.5))
        child_a, child_b = simulated_binary_crossover(father, mother, eta)
        self.assertTrue(all(x == -1 for x in child_a[0]))
        self.assertTrue(all(x == 1 for x in child_b[0]))
        np.random.random = self.backup_np_random_random

    def test_single_point_binary_crossover(self):
        father = [
            np.array([[0, 0], [0, 0]])
        ]
        mother = [
            np.array([[1, 1], [1, 1]])
        ]
        np.random.randint = MagicMock(return_value=1)
        child_a, child_b = single_point_binary_crossover(father, mother)
        self.assertTrue(np.array_equal(np.array([[1, 1], [1, 0]]), child_a[0]))
        self.assertTrue(np.array_equal(np.array([[0, 0], [0, 1]]), child_b[0]))
        np.random.randint = self.backup_random_random_int

    def test_couple_crossover_same_parent(self):
        # Arrange
        father = [
            np.array([[0, 0], [0, 0]])
        ]
        # Act
        children_random = random_crossover(father, father)

        # Assert
        for i in range(len(father)):
            np.testing.assert_array_equal(father[i], children_random[0][i])

    def test_random_crossover(self):
        parent_model_genetic = [np.array([[0, 0], [0, 0]])]
        mother_model_genetic = [np.array([[1, 1], [1, 1]])]
        children = random_crossover(parent_model_genetic, mother_model_genetic)
        for child in children:
            for layer_index in range(len(parent_model_genetic)):
                parent_array = parent_model_genetic[layer_index]
                mother_array = mother_model_genetic[layer_index]
                child_array = child[layer_index]
                equal_to_father = np.sum(np.where(parent_array == child_array, 1, 0))
                equal_to_mother = np.sum(np.where(mother_array == child_array, 1, 0))
                self.assertEqual(np.prod(parent_array.shape), equal_to_father + equal_to_mother)
