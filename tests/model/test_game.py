import unittest

from model.game import Game
from model.point import Point


class TestGame(unittest.TestCase):

    def test_move_left_down(self):
        # Arrange
        size = 5
        snake_start = [[2, 2], [3, 2], [4, 2]]
        snake_left = [[1, 2], [2, 2], [3, 2]]
        snake_down = [[1, 3], [1, 2], [2, 2]]
        apple = [4, 4]
        game = Game(size, snake_start, apple)
        # Act left
        game.move(Game.LEFT)
        self.assertEqual(Point.ints_to_points(snake_left), game.snake)
        # Act down
        game.move(Game.DOWN)
        self.assertEqual(Point.ints_to_points(snake_down), game.snake)

    def test_move_right_up(self):
        # Arrange
        size = 5
        snake_start = [[3, 2], [2, 2], [1, 2]]
        snake_right = [[4, 2], [3, 2], [2, 2]]
        snake_up = [[4, 1], [4, 2], [3, 2]]
        apple = [4, 4]
        game = Game(size, snake_start, apple)
        # Act right
        game.move(Game.RIGHT)
        self.assertEqual(Point.ints_to_points(snake_right), game.snake)
        # Act up
        game.move(Game.UP)
        self.assertEqual(Point.ints_to_points(snake_up), game.snake)

    def test_move_take_apple(self):
        # Arrange
        size = 5
        snake_start = [[3, 2], [2, 2], [1, 2]]
        snake_right = [[4, 2], [3, 2], [2, 2], [1,2]]
        apple = [4, 2]
        game = Game(size, snake_start, apple)
        # Act right and eat the apple
        game.move(Game.RIGHT)
        self.assertEqual(Point.ints_to_points(snake_right), game.snake)

    def test_is_valid_input(self):
        size = 2
        snake_start = [[13, 2], [2, 2], [1, 2]]
        apple = [4, 2]

        # Size < 3
        self.assertRaises(ValueError, lambda: Game(size, snake_start, apple))

        size = 10
        # Snake out of boundaries
        self.assertRaises(ValueError, lambda: Game(size, snake_start, apple))

        # Apple out of boundaries
        size = 50
        apple = [51, 2]
        self.assertRaises(ValueError, lambda: Game(size, snake_start, apple))
