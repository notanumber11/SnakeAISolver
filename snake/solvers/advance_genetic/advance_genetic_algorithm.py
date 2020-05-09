from typing import List
import numpy as np
import tensorflow as tf
from game.game import Game
from game.game_status import GameStatus
from solvers.advance_genetic.advanced_genetic_model_evaluated import AdvanceModelGeneticEvaluated
from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator


class AdvanceGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, layers_size: List[int]):
        super().__init__(layers_size)
        self.advance_training_data_generator = AdvanceTrainingDataGenerator()
        if layers_size[-1] < 2:
            raise ValueError("AdvanceGeneticAlgorithm expects the output layer to be >1 since it uses classification {}"
                             .format(layers_size))


    def evaluate_model(self, game_status_seeds: List[GameStatus], model, model_genetic) -> AdvanceModelGeneticEvaluated:
        self._set_model_weights(self.model, model_genetic)
        games = []
        for game_status in game_status_seeds:
            # Play one game
            game = self.play_one_game(game_status, model)
            games.append(game)
        # Average the results from all games
        return AdvanceModelGeneticEvaluated(games, model_genetic)

    def play_one_game(self, game_status_seed: GameStatus, model):
        game_statuses = []
        movements_left = game_status_seed.get_number_of_holes()
        while game_status_seed.is_valid_game() and movements_left > 0:
            _input = [self.advance_training_data_generator.get_input_from_game_status(game_status_seed)]
            _dir = self.get_best_movement(_input, model)
            new_game_status = game_status_seed.move(_dir)
            game_statuses.append(new_game_status)
            # Continue iteration
            game_status_seed = new_game_status
            movements_left -= 1

        # The game was in a loop
        is_loop = True if movements_left == 0 else False
        game = Game(game_statuses, is_loop)
        return game


    def get_best_movement(self, _input, model):
        tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        test_predictions = model.predict([_input])
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]