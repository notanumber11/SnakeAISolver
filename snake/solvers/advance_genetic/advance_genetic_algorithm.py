from typing import List
import numpy as np
import tensorflow as tf
from game.game import Game
from game.game_status import GameStatus
from solvers.advance_genetic.advance_genetic_model_evaluated import AdvanceModelGeneticEvaluated
from solvers.basic_genetic.genetic_algorithm import GeneticAlgorithm
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator


class AdvanceGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, layers_size: List[int]):
        super().__init__(layers_size)
        self.advance_training_data_generator = AdvanceTrainingDataGenerator()
        if layers_size[-1] < 2:
            raise ValueError("AdvanceGeneticAlgorithm expects the output layer to be >1 since it uses classification {}"
                             .format(layers_size))

    def evaluate_population(self, population_genetic: List, game_status_seeds: List[GameStatus]) \
            -> List[AdvanceModelGeneticEvaluated]:
        """
        :param population_genetic: List of population size containing model_genetics. Each model_genetic is a list
        of weights for each layer in the neural network.
        :param game_status_seeds: List of GameStatus that will be used for the initial state of the games that will
        be played by each model_genetic.
        :return: List sorted descending based on fitness.
        """
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
            # Play one game
            game = self.play_one_game(game_status, model)
            games.append(game)
        # Average the results from all games
        return AdvanceModelGeneticEvaluated(games, model_genetic)

    def play_one_game(self, current_game_status: GameStatus, model):
        game_statuses = [current_game_status]
        movements_left = current_game_status.get_number_of_holes()
        while current_game_status.is_valid_game() and movements_left > 0:
            _input = [self.advance_training_data_generator.get_input_from_game_status(current_game_status)]
            _dir = self.get_best_movement(_input, model)
            new_game_status = current_game_status.move(_dir)
            game_statuses.append(new_game_status)
            # Continue iteration
            movements_left -= 1
            if current_game_status.apple != new_game_status.apple:
                movements_left = current_game_status.get_number_of_holes()
            current_game_status = new_game_status

        # The game was in a loop
        is_loop = True if movements_left == 0 else False
        game = Game(game_statuses, is_loop)
        return game

    def get_best_movement(self, _input, model):
        tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        test_predictions = model.predict(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        result = GameStatus.DIRS[max_index]
        return result
