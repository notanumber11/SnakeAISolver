import numpy as np

from game.game_status import GameStatus
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.classification.short_vision import ShortVision
from solvers.solver import Solver


class BasicGeneticSolver(Solver):

    def __init__(self, path_model=None):
        if path_model == None:
            path_model = r"models/basic_genetic/pop=1000_sel=0.1_mut_0.05_it_10000_games_1_game_size_6/46_____completion_36.0_36.0___1.00_____movements_326.0"
            print("path model is: " + path_model)
        self.model = data_utils.load_model(path_model)
        self.data_provider = ShortVision()

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        max_movements = game_status.size ** 4
        while game_status.is_valid_game() and max_movements > 0:
            max_movements -= 1
            _input = [self.data_provider.get_input_from_game_status(game_status)]
            dir = self.get_best_movement(_input, self.model)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        self.finished()
        return game_statuses

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        result = GameStatus.DIRS[max_index]
        return result
