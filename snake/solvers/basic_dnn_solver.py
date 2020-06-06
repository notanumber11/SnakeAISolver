import numpy as np

from game.game_status import GameStatus
from solvers.training_data_generators import data_utils
from solvers.solver import Solver
from solvers.training_data_generators.regression.short_vision_regression import ShortVisionRegression


class BasicDnnSolver(Solver):

    def __init__(self):
        path_model = r"models/basic_dnn/model_basic_dnn_mse_9.16E-03_samples_10000"
        self.model = data_utils.load_model(path_model)
        self.data_provider = ShortVisionRegression()

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        counter = game_status.size ** 4
        while game_status.is_valid_game() and counter > 0:
            counter -= 1
            inputs = self.data_provider.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        self.finished()
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.__call__(np.array(inputs))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

