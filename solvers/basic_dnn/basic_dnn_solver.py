import numpy as np

from model.game_status import GameStatus
import solvers.basic_dnn.basic_dnn as basic_dnn
import solvers.basic_dnn.training_data_generator


class BasicDnnSolver:

    def __init__(self):
        path_model = r"C:\Users\Denis\Desktop\SnakePython\data\basic_dnn\mode_basic_dnn_mse_7.12E-03_samples_10000"
        path_model = r"C:\Users\Denis\Desktop\SnakePython\data\basic_genetic\pop=1000_sel=0.1_mut_0.01_it_50_games_1\18_iterations_snake_length_5_reward_1.2999999999999998_movements_6"
        self.model = basic_dnn.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        while game_status.is_valid_game():
            inputs = solvers.basic_dnn.training_data_generator.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("dnn game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.predict(inputs).flatten()
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def __str__(self):
        return "BasicDnnSolver"
