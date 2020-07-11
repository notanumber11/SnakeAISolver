import numpy as np

from game.game_status import GameStatus
import solvers.reward_based_dnn_solver
from solvers.solver import Solver
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.classification.distance_vision import DistanceVision


class DistanceVisionGeneticSolverWithFallback(Solver):

    def __init__(self, path_model=None):
        super().__init__()
        best_path_model = "models/advance_genetic/training=distance_vision_pop=1000_sel=0.1_mut_0.05_it_10000_game_size_8/624.0_apples_62.0_size_64.0_movements_1342.0/"
        self.path_model = path_model if path_model is not None else best_path_model
        self.data_provider = DistanceVision()
        self.fallback_solver = solvers.reward_based_dnn_solver.RewardBasedDnnSolver()

    def solve(self, prev: GameStatus):
        model = data_utils.load_model(self.path_model)
        game_statuses = [prev]

        while prev.is_valid_game():
            input = self.data_provider.get_input_from_game_status(prev)
            _dir = self.get_best_movement(input, model)

            # If we are not able to move to a valid position we use the fallback dnn
            if not prev.can_move_to_dir(_dir):
                _dir = self.fallback_solver.get_best_movement(
                    self.fallback_solver.data_provider.get_input_from_game_status(prev), self.fallback_solver.model)
            new = prev.move(_dir)
            game_statuses.append(new)

            # Fallback to dnn if we are stack in a loop
            if self.is_loop(prev, new):
                new_game_statuses = self.get_one_apple(prev)
                if not new_game_statuses:
                    break
                game_statuses += new_game_statuses
                self.movements_left = game_statuses[-1].get_movements_left()
            prev = game_statuses[-1]

        self.finished()
        return game_statuses

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        return GameStatus.DIRS[max_index]

    def get_one_apple(self, prev: GameStatus):
        game_statuses = []
        apple = prev.apple
        while prev.is_valid_game():
            _dir = self.fallback_solver.get_best_movement(
                self.fallback_solver.data_provider.get_input_from_game_status(prev), self.fallback_solver.model)
            new = prev.move(_dir)
            game_statuses.append(new)
            if new.apple != apple:
                return game_statuses
            if self.fallback_solver.is_loop(prev, new):
                return []
            prev = new
        return game_statuses
