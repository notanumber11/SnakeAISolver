import numpy as np

import solvers.distance_vision_genetic_solver
from game.game_status import GameStatus
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.regression.short_vision_regression import ShortVisionRegression


class RewardBasedDnnSolver(solvers.distance_vision_genetic_solver.DistanceVisionGeneticSolver):

    def __init__(self, path_model=None):
        super().__init__(path_model)
        best_path_model = r"models/basic_dnn/model_basic_dnn_mse_9.16E-03_samples_10000"
        self.path_model = path_model if path_model is not None else best_path_model
        self.data_provider = ShortVisionRegression()
        self.model = data_utils.load_model(self.path_model)

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]
