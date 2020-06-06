import numpy as np

from game.game_status import GameStatus
from solvers.data_providers import basic_training_data_generator, data_utils


class BasicGeneticSolver:

    def __init__(self, path_model=None):
        if path_model == None:
            path_model = r"models/basic_genetic/52_iterations_snake_length_36.0_movements_279.0reward_20.49"
            print("path model is: " + path_model)
        self.model = data_utils.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        max_movements = game_status.size ** 4
        while game_status.is_valid_game() and max_movements > 0:
            max_movements -= 1
            inputs = basic_training_data_generator.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("basic genetic game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.__call__(np.array(inputs))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]
