import numpy as np

from game.game_status import GameStatus
from solvers.training import basic_training_data_generator, training_utils


class BasicGeneticSolver:

    def __init__(self):
        path_model = r"..\data\basic_genetic\success_genetic\31_iterations_snake_length_24.0_movements_177.0reward_13.899999999999974_"
        path_model = r"..\data\basic_genetic\success_genetic\33_iterations_snake_length_26.0_movements_180.0reward_17.49999999999999_"
        path_model = r"..\data\basic_genetic\success_genetic\s3\52_iterations_snake_length_36.0_movements_279.0reward_20.49"
        self.model = training_utils.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        max_movements = game_status.size**4
        while game_status.is_valid_game() and max_movements > 0:
            max_movements -= 1
            inputs = basic_training_data_generator.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("basic genetic game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.predict(inputs).flatten()
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]