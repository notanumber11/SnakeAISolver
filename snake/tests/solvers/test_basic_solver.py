import unittest

from game.game_status import GameStatus
from solvers.survival_random_solver import SurvivalRandomSolver


class TestGame(unittest.TestCase):

    def test_next_dir(self):
        # Create a game
        size = 10
        snake_start = [[2, 2], [3, 2], [4, 2], [5, 2]]
        apple = [1, 2]
        game = GameStatus(size, snake_start, apple)
        # The right direction is left to pick the apple
        expected_dir = GameStatus.LEFT
        # Solve the game
        basic_solver = SurvivalRandomSolver()
        next_dir = basic_solver.get_best_movement(game)
        # self.assertEqual(expected_dir, next_dir)
