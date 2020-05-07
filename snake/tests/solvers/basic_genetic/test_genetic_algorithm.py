import unittest

import numpy
import numpy as np
from numpy.testing import assert_raises

from game.game_status import GameStatus
from solvers.training import basic_training_data_generator
from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm
from game.game_seed_creator import create_default_game_seed
import timeit


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
        # Assert
        self.assertEqual(population_size, len(population_genetic))
        self.assertTrue(len(population_genetic[0]) == len(self.layers_size) - 1)
        self.assertEqual((9, 125), layer_0_genetic_size)

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
        # All values are between -1 and 1
        for model_genetic in population_genetic:
            self.assertTrue(((-1 <= model_genetic[0]) & (model_genetic[0] <= 1)).all())

    def test_evaluate_population(self):
        # Arrange
        population_size = 3
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        # Act
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)
        # Assert
        self.assertEqual(len(population_with_fitness), population_size)

    def test_evaluate_model(self):
        population_size = 20
        game_seeds_size = 1
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        game_seeds = [create_default_game_seed() for i in range(game_seeds_size)]
        random_fitness = []
        fix_fitness = []
        for model_genetic in population_genetic:
            for layer_index in range(len(model_genetic)):
                model_genetic[layer_index] = np.random.uniform(low=-1, high=1, size=model_genetic[layer_index].shape)
                model_evaluated = self.ga.evaluate_model(game_seeds, self.ga.model, model_genetic)
                random_fitness.append(model_evaluated.fitness())
                model_genetic[layer_index] = np.zeros(model_genetic[layer_index].shape)
                model_evaluated = self.ga.evaluate_model(game_seeds, self.ga.model, model_genetic)
                fix_fitness.append(model_evaluated.fitness())
        self.assertFalse(all(x == random_fitness[0] for x in random_fitness))
        self.assertFalse(any(x != fix_fitness[0] for x in fix_fitness))

    def test_selection_sort_population(self):
        # Arrange
        population_size = 10
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)
        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x.fitness(), reverse=True)
        threshold = 0.2
        # Act
        self.assertNotEqual(population_with_fitness, sorted_population_with_fitness)
        self.ga.selection(threshold, population_with_fitness)
        self.assertEqual(population_with_fitness, sorted_population_with_fitness)

    def test_selection(self):
        # Arrange
        population_size = 10
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)
        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x.fitness(), reverse=True)
        threshold = 0.4
        number_of_top_performers = int(threshold * population_size)
        top_performers = sorted_population_with_fitness[0:number_of_top_performers]
        worst_performers = sorted_population_with_fitness[number_of_top_performers:]

        # Act
        selected_pairs = self.ga.selection(threshold, population_with_fitness)
        selected_individuals = [individual for pair in selected_pairs for individual in pair]

        # Assert
        self.assertEqual(len(selected_pairs) * 2, population_size)
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
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)

        for i in range(population_size):
            population_with_fitness[i].snake_length = 10

        # One performer has significantly more fitness that the others
        population_with_fitness[-1].snake_length = population_size ** 10

        top = population_with_fitness[-1]
        worst = population_with_fitness[0]
        boy, girl = self.ga._pair(population_with_fitness, population_size ** 10)
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
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        population_with_fitness = self.ga.evaluate_population(population_genetic, game_statuses)

        # Each individual has 10% chances of being chosen since all of them share the same fitness
        for i in range(population_size):
            population_with_fitness[i].snake_length = fitness
            total_fitness += population_with_fitness[i].fitness()

        # If we obtain 100 elements the probability of one individual to be selected is ps = 1-(1/10)^100 ~= 1
        result = []
        for i in range(number_of_pairs_to_generate):
            pair = self.ga._pair(population_with_fitness, total_fitness)
            result += pair

        self.assertEqual(number_of_pairs_to_generate * 2, len(result))
        self.assertTrue(all(el in result for el in population_with_fitness))

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
        children_fix = self.ga._fix_crossover(parent_model_genetic, mother_model_genetic, even_masks, odd_masks)
        children_random = self.ga._random_crossover(parent_model_genetic, mother_model_genetic)

        # Assert
        for layer_index in range(len(parent_model_genetic)):
            np.testing.assert_array_equal(parent_model_genetic[layer_index], mother_model_genetic[layer_index])
            np.testing.assert_array_equal(parent_model_genetic[layer_index], children_fix[0][layer_index])
            np.testing.assert_array_equal(parent_model_genetic[layer_index], children_random[0][layer_index])

    def test_fix_crossover(self):
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
        children = self.ga._fix_crossover(parent_model_genetic, mother_model_genetic, even_masks, odd_masks)

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

    def test_random_crossover(self):
        population_size = 2
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        parent_model_genetic = population_genetic[0]
        mother_model_genetic = population_genetic[1]
        children = self.ga._random_crossover(parent_model_genetic, mother_model_genetic)

        for child in children:
            for layer_index in range(len(parent_model_genetic)):
                parent_array = parent_model_genetic[layer_index]
                mother_array = mother_model_genetic[layer_index]
                child_array = child[layer_index]
                equal_to_father = np.sum(np.where(parent_array == child_array, 1, 0))
                equal_to_mother = np.sum(np.where(mother_array == child_array, 1, 0))
                self.assertEqual(np.prod(parent_array.shape), equal_to_father + equal_to_mother)
                return
        # Ensure that we execute the loop
        self.assertTrue(False)

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
        for layer_index in range(len(model_genetic)):
            np.testing.assert_array_equal(model_genetic[layer_index], np.ones(size))
        # mutate a portion
        self.ga.mutation(0.05, model_genetic)
        equal_count = 0
        for layer_index in range(len(model_genetic)):
            for i in range(size):
                if model_genetic[layer_index][i] == 1:
                    equal_count += 1
        self.assertGreaterEqual(equal_count, 5 * mutation_rate * size)
        self.assertLessEqual(equal_count, size * (1 - mutation_rate / 5))

    def test_execute_iteration_with_single_element(self):
        games_to_play = 2
        population_size = 6
        # Create a population genetic with the same element repeated
        population_genetic = self.ga.get_initial_population_genetic(1) * population_size
        selection_threshold = 0.5
        # Do not mutate
        mutation_rate = 0
        children, evaluation_result = self.ga.execute_iteration(population_genetic, games_to_play, selection_threshold,
                                                                mutation_rate)
        self.assertEqual(len(children), population_size)
        for child in children:
            for layer_index in range(len(child)):
                np.testing.assert_array_equal(child[layer_index], population_genetic[0][layer_index])

    def test_execute_iteration(self):
        games_to_play = 1
        population_size = 10
        # Create a population genetic with the same element
        population_genetic = self.ga.get_initial_population_genetic(population_size)
        selection_threshold = 0.2
        mutation_rate = 0.2
        children, evaluation_result = self.ga.execute_iteration(population_genetic, games_to_play, selection_threshold,
                                                                mutation_rate)
        # self.assertEqual(len(children), population_size)
        # for child in children:
        #     for layer_index in range(len(child)):
        #         self.assertTrue(((-1 <= child[layer_index]) & (child[layer_index] <= 1)).all())

    def test_evaluate_population_performance(self):
        layers_size = [9, 125, 1]
        population_size = 10
        ga = GeneticAlgorithm(layers_size)
        game_statuses = [create_default_game_seed() for i in range(1)]
        population_genetic = ga.get_initial_population_genetic(population_size)
        result = timeit.timeit(lambda: ga.evaluate_population(population_genetic, game_statuses), number=1)
        print(result)

    def test_get_best_movement_performance(self):
        # Initialize game with random weights
        population_genetic = self.ga.get_initial_population_genetic(1)
        game_status = create_default_game_seed()
        inputs = basic_training_data_generator.get_input_from_game_status(game_status)
        result = timeit.timeit(lambda: self.ga.get_best_movement(inputs, self.ga.model), number=10)
        print(result)

    #
    # def test_only_modify_hidden_layer(self):
    #     pass
