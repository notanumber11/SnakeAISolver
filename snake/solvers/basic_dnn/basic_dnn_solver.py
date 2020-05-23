import numpy as np

from game.game_status import GameStatus
import solvers.training.basic_training_data_generator
from solvers.training import training_utils


class BasicDnnSolver:

    def __init__(self):
        path_model = r"models/basic_dnn/mode_basic_dnn_mse_7.12E-03_samples_10000"
        self.model = training_utils.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        counter = game_status.size**4
        while game_status.is_valid_game() and counter > 0:
            counter -= 1
            inputs = solvers.training.basic_training_data_generator.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("dnn game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.__call__(np.array(inputs))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def __str__(self):
        return "BasicDnnSolver"
