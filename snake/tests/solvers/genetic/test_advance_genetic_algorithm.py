from unittest.mock import MagicMock

from game import game_seed_creator
from game.game_seed_creator import create_default_game_seed, create_random_game_seed
from solvers.genetic.advance_genetic_algorithm import AdvanceGeneticAlgorithm
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator
from tests.solvers.genetic.test_genetic_algorithm import GeneticAlgorithmTest


class TestAdvanceGeneticAlgorithm(GeneticAlgorithmTest):

    def setUp(self):
        self.layers_size = [28, 20, 12, 4]
        self.ga = AdvanceGeneticAlgorithm(self.layers_size, AdvanceTrainingDataGenerator())

    def test_max_number_of_movements(self):
        game_status = game_seed_creator.create_default_game_seed()
        game_status.is_valid_game = MagicMock(return_value=True)
        game_status.move = MagicMock(return_value=game_status)
        game = self.ga.play_one_game(game_status, self.ga.model, self.ga.training_generator)
        self.assertTrue(game.was_stack_in_loop)
        self.assertEqual(len(game.game_statuses), game_status.get_number_of_holes() + 1)

    def test_evaluate_best_movement_with_different_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
        for i in range(100):
            model_genetic = self.ga.get_initial_population_genetic(1)[0]
            self.ga._set_model_weights(self.ga.model, model_genetic)
            predictions.append(self.ga.get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the weights are different
        self.assertFalse(all(x == predictions[0] for x in predictions))

    def test_evaluate_best_movement_with_same_model(self):
        game_status = create_default_game_seed()
        predictions = []
        _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
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
            _input = [self.ga.training_generator.get_input_from_game_status(game_status)]
            predictions.append(self.ga.get_best_movement(_input, self.ga.model))
        # Not all predictions are the same since the input is different
        self.assertFalse(all(x == predictions[0] for x in predictions))
        prediction_set = set(predictions)
        print(prediction_set)


    def test_absolute_distances(self):
        pass

    def direction_as_input(self):
        pass

    def use_hyperparemeters_as_setting_and_track_experiments(self):
        pass