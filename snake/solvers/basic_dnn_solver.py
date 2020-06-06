import numpy as np

import solvers.data_providers.dnn_training_data_generator
from game.game_status import GameStatus
from solvers.data_providers import data_utils
from solvers.solver import Solver


class BasicDnnSolver(Solver):

    def __init__(self):
        path_model = r"models/basic_dnn/model_basic_dnn_mse_8.20E-03_samples_10000"
        self.model = data_utils.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        counter = game_status.size ** 4
        while game_status.is_valid_game() and counter > 0:
            counter -= 1
            inputs = solvers.data_providers.dnn_training_data_generator.get_inputs_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        self.finished()
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.__call__(np.array(inputs))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

