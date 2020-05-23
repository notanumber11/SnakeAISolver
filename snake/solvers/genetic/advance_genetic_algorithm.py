from typing import List

import numpy as np
import tensorflow as tf

from game.game_status import GameStatus
from solvers.genetic.advance_genetic_solver import AdvanceGeneticSolver
from solvers.genetic.genetic_algorithm import GeneticAlgorithm
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator
from utils import aws_snake_utils


class AdvanceGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, layers_size: List[int]):
        super().__init__(layers_size)
        self.training_generator = AdvanceTrainingDataGenerator()
        if layers_size[-1] < 2:
            raise ValueError("AdvanceGeneticAlgorithm expects the output layer to be >1 since it uses classification {}"
                             .format(layers_size))

    def build_model(self):
        model = super().build_model()
        tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        return model

    def get_best_movement(self, _input, model):
        test_predictions = model.__call__(np.array(_input))
        max_index = np.argmax(test_predictions[0])
        result = GameStatus.DIRS[max_index]
        return result

    def show_current_best_model(self, iteration, path, game_size):
        if aws_snake_utils.is_local_run() and iteration % 25 == 0:
            from gui.gui_starter import show_solver
            show_solver(AdvanceGeneticSolver(path), game_size, 3, 6)