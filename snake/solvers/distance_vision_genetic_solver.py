import numpy as np

from game.game_status import GameStatus
from solvers.solver import Solver
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.classification.distance_vision import DistanceVision


class DistanceVisionGeneticSolver(Solver):

    def __init__(self, path_model=None):
        super().__init__()
        best_path_model = "models/advance_genetic/training=distance_vision_pop=1000_sel=0.1_mut_0.05_it_10000_game_size_8/624.0_apples_62.0_size_64.0_movements_1342.0/"
        # best_path_model = "models/advance_genetic/pop=1000_sel=0.1_mut_0.06_it_10000_games_1_game_size_8/201.00_iterations_fitness_79947.64_snake_length_64.00_movements_493.00"
        # best_path_model = "models/advance_genetic/pop=1000_sel=0.1_mut_0.05_it_10000_games_1_game_size_16/176_____completion_256.0_256.0___1.00_____movements_12396.0"
        self.path_model = path_model if path_model is not None else best_path_model
        self.data_provider = DistanceVision()
        self.model = data_utils.load_model(self.path_model)

    def solve(self, prev: GameStatus):
        game_statuses = [prev]
        while prev.is_valid_game():
            input = self.data_provider.get_input_from_game_status(prev)
            _dir = self.get_best_movement(input, self.model)
            new = prev.move(_dir)
            game_statuses.append(new)
            if self.is_loop(prev, new):
                break
            prev = new
        self.finished()
        return game_statuses

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        return GameStatus.DIRS[max_index]
