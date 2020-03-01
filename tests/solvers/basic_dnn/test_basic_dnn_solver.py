import unittest

from model import game_seed_creator
from solvers.basic_dnn.basic_dnn_solver import BasicDnnSolver


class TestBasicDnnSolver(unittest.TestCase):

    def test_next_dir(self):
        dnn_solver = BasicDnnSolver()
        game_status = game_seed_creator.create_default_game_seed()
        inputs = dnn_solver.get_input_from_game_status(game_status)
        input = [1, 0, 0, 0, 1, 1, 1, 0, -0.3239560407317998]
        self.assertEqual(input, inputs[0])
        self.assertEqual(len(inputs), 4)
        dnn_solver.get_best_movement(inputs)