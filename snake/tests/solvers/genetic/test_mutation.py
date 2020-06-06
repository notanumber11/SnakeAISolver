import unittest

import numpy as np

from solvers.genetic.mutation import gaussian_mutation, uniform_mutation


class TestMutation(unittest.TestCase):

    def test_gaussian_mutation(self):
        mu = 0
        sigma = 1
        mutation_rate = 1
        model_genetic = [
            np.array([float(0) for x in range(1000)])
        ]
        gaussian_mutation(model_genetic, mutation_rate=mutation_rate, mu=mu, sigma=sigma)
        self.assertTrue(abs(mu - np.mean(model_genetic[0])) < 0.2)

    def test_uniform_mutation(self):
        size = 100
        mutation_rate = 0.05
        model_genetic = [np.ones(size)]
        # mutate 0
        uniform_mutation(model_genetic, 0)

        # mutate a portion
        uniform_mutation(model_genetic, mutation_rate)
        equal_count = 0
        for layer_index in range(len(model_genetic)):
            for i in range(size):
                if model_genetic[layer_index][i] == 1:
                    equal_count += 1
        self.assertGreaterEqual(equal_count, 70)
        self.assertLessEqual(equal_count, 99)

    def test_zero_mutation_rate(self):
        mutation_rate = 0
        size = 1000
        model_genetic = [np.ones(size)]
        gaussian_mutation(model_genetic, mutation_rate=mutation_rate, mu=0, sigma=1)
        for layer_index in range(len(model_genetic)):
            np.testing.assert_array_equal(model_genetic[layer_index], np.ones(size))
        uniform_mutation(model_genetic, mutation_rate)
        for layer_index in range(len(model_genetic)):
            np.testing.assert_array_equal(model_genetic[layer_index], np.ones(size))
