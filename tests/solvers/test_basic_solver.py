import unittest

from model.game import Game
from solvers.basic_solver import BasicSolver


class TestGame(unittest.TestCase):

    def test_next_dir(self):
        # Create a game
        size = 10
        snake_start = [[2, 2], [3, 2], [4, 2], [5, 2]]
        apple = [1, 2]
        game = Game(size, snake_start, apple)
        # The right direction is left to pick the apple
        expected_dir = Game.LEFT
        # Solve the game
        basicSolver = BasicSolver()
        next_dir = basicSolver.next_dir(game)
        self.assertEqual(expected_dir, next_dir)