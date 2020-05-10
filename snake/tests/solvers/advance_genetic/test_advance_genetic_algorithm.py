import timeit
from unittest.mock import MagicMock

from game import game_seed_creator
from game.game_seed_creator import create_default_game_seed, create_random_game_seed
from solvers.advance_genetic.advance_genetic_algorithm import AdvanceGeneticAlgorithm
from tests.solvers.basic_genetic.test_genetic_algorithm import GeneticAlgorithmTest
import numpy as np


class TestAdvanceGeneticAlgorithm(GeneticAlgorithmTest):

    def setUp(self):
        self.layers_size = [28, 20, 12, 4]
        self.ga = AdvanceGeneticAlgorithm(self.layers_size)

    def test_max_number_of_movements(self):
        game_status = game_seed_creator.create_default_game_seed()
        game_status.is_valid_game = MagicMock(return_value=True)
        game_status.move = MagicMock(return_value=game_status)
        game = self.ga.play_one_game(game_status, self.ga.model)
        self.assertTrue(game.was_stack_in_loop)
        self.assertEqual(len(game.game_statuses), game_status.get_number_of_holes() + 1)

    def parallel_processing_benchmark(self):
        size = 1000
        # Create 100 different inputs
        inputs = [self.ga.advance_training_data_generator.get_input_from_game_status(create_default_game_seed())
                  for i in range(size)]

        # warm up
        for i in range(10):
            self.ga.get_best_movement(inputs, self.ga.model)

        result = timeit.timeit(lambda: self.ga.get_best_movement(inputs, self.ga.model), number=1)
        print("{} in parallel took {} seconds".format(size, result))

        result = timeit.timeit(lambda: self.ga.get_best_movement([inputs[0]], self.ga.model), number=size)
        print("{} in sequential took {} seconds".format(size, result))


    def test_evaluate_best_movement_with_different_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.advance_training_data_generator.get_input_from_game_status(game_status)]
        for i in range(100):
            model_genetic = self.ga.get_initial_population_genetic(1)[0]
            self.ga._set_model_weights(self.ga.model, model_genetic)
            predictions.append(self.ga.get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the weights are different
        self.assertFalse(all(x == predictions[0] for x in predictions))

    def test_evaluate_best_movement_with_same_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.advance_training_data_generator.get_input_from_game_status(game_status)]
        model_genetic = self.ga.get_initial_population_genetic(1)[0]
        self.ga._set_model_weights(self.ga.model, model_genetic)
        for i in range(100):
            predictions.append(self.ga.get_best_movement(_input, self.ga.model))
        # All predictions are the same since the weights are the same
        self.assertFalse(any(x != predictions[0] for x in predictions))

    def test_evaluate_best_movement_with_same_model_different_input(self):
        predictions = []
        model_genetic = self.ga.get_initial_population_genetic(1)[0]
        self.ga._set_model_weights(self.ga.model, model_genetic)
        for i in range(100):
            game_status = create_random_game_seed(6, 2)
            _input = [self.ga.advance_training_data_generator.get_input_from_game_status(game_status)]
            predictions.append(self.ga.get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the input is different
        self.assertFalse(all(x == predictions[0] for x in predictions))
        prediction_set = set(predictions)
        print(prediction_set)
