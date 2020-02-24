import unittest

from model.game_status import GameStatus
from solvers.basic_solver import BasicSolver


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
        basic_solver = BasicSolver()
        next_dir = basic_solver.next_dir(game)
        # self.assertEqual(expected_dir, next_dir)
