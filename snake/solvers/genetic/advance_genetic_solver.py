import os

import numpy as np

from game.game_status import GameStatus
from solvers.training import training_utils
import solvers.training.advance_training_data_generator
from solvers.training.full_body_vision_training_data_generator import FullBodyVisionTrainingDataGenerator


class AdvanceGeneticSolver:

    def __init__(self, path_model=None):
        if path_model is None:
            path_model = r"models/advance_genetic/1682.00_iterations_fitness_7490.47_snake_length_26.00_mov"
        print("advanced path model is: " + path_model)
        self.model = training_utils.load_model(path_model)
        self.ag = FullBodyVisionTrainingDataGenerator()


    def solve(self, current_game_status: GameStatus):
        game_statuses = [current_game_status]
        movements_left = current_game_status.size**2 * 2
        while current_game_status.is_valid_game() and movements_left > 0:
            movements_left -= 1
            input = [self.ag.get_input_from_game_status(current_game_status)]
            _dir = self.get_best_movement(input, self.model)
            new_game_status = current_game_status.move(_dir)
            game_statuses.append(new_game_status)
            if current_game_status.apple != new_game_status.apple:
                movements_left = current_game_status.size**2 * 2
            if movements_left == 0:
                print("loop time !")
            current_game_status = new_game_status
        print("advance genetic game solved")
        return game_statuses

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        result = GameStatus.DIRS[max_index]
        return result
