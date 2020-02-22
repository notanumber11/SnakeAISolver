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
        basic_solver = BasicSolver()
        next_dir = basic_solver.next_dir(game)
        self.assertEqual(expected_dir, next_dir)


        one = [1, 2, 3]
        two = [99, 0]
        three = one + two
        print(three)