import numpy as np

from model.game_status import GameStatus
from solvers.basic_dnn import constants, dnn_orquestrator


class BasicDnnSolver:

    def __init__(self):
        self.model = dnn_orquestrator.load_basic_dnn_model()

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        while game_status.is_valid_game():
            inputs = self.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("dnn game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.predict(inputs).flatten()
        max_index = np.where(test_predictions == np.amax(test_predictions))[0][0]
        return GameStatus.DIRS[max_index]

    def get_input_from_game_status(self, game_status: GameStatus):
        """
        The goal of this method is to create 4 inputs (one per direction) with the following data
        ["up", "down", "left", "right", "up available", "down available", "left available", "right available", "angle to apple"]
        Example for up:
        [1, 0, 0, 0, 1, 1, 0, 0, 0.8]
        """
        angle = constants.normalize_rad_angle(game_status.get_angle(game_status.apple, game_status.head))
        available = [1 if game_status.can_move_to_dir(d) else 0 for d in GameStatus.DIRS]
        inputs = []
        for i in range(len(GameStatus.DIRS)):
            input = [0] * 4
            input[i] = 1
            inputs.append(input)
        for input in inputs:
            input += available
            input.append(angle)
        return inputs

    def __str__(self):
        return "BasicDnnSolver"