import tensorflow as tf
import numpy as np

from model.game_status import GameStatus
from solvers.basic_dnn import training_data_generator


class BasicGeneticSolver():

    def __init__(self):
        path_model = r"C:\Users\Denis\Desktop\SnakePython\data\success_genetic\pop=1000_sel=0.1_mut_0.01_it_50_games_1\33_iterations_snake_length_26.0_movements_180.0reward_17.49999999999999_"
        self.model = self.load_model(path_model)

    def solve(self, game_status: GameStatus):
        game_statuses = [game_status]
        counter = 1000
        while game_status.is_valid_game() and counter > 0:
            counter -= 1
            if counter == 0:
                print("Loop !: {}".format(1000-counter))
            inputs = training_data_generator.get_input_from_game_status(game_status)
            dir = self.get_best_movement(inputs)
            game_status = game_status.move(dir)
            game_statuses.append(game_status)
        print("dnn game solved")
        return game_statuses

    def get_best_movement(self, inputs):
        test_predictions = self.model.predict(inputs).flatten()
        max_index = np.argmax(test_predictions)
        return GameStatus.DIRS[max_index]

    def load_model(self, path: str):
        path = path.replace("/", "\\")
        new_model = tf.keras.models.load_model(path)
        return new_model