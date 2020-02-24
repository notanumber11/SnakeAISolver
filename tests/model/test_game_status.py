import unittest

from model.game_status import GameStatus
from model.point import Point


class TestGame(unittest.TestCase):

    def test_move_left_down(self):
        # Arrange
        size = 5
        snake_start = [[2, 2], [3, 2], [4, 2]]
        snake_left = [[1, 2], [2, 2], [3, 2]]
        snake_right = [[3, 2], [2, 2], [3, 2]]
        snake_down = [[2, 3], [2, 2], [3, 2]]
        snake_up = [[2, 1], [2, 2], [3, 2]]
        apple = [4, 4]
        game = GameStatus(size, snake_start, apple)
        # Act left
        new_game = game.move(GameStatus.LEFT)
        self.assertEqual(Point.ints_to_points(snake_left), new_game.snake)
        # Act right
        new_game = game.move(GameStatus.RIGHT)
        self.assertEqual(Point.ints_to_points(snake_right), new_game.snake)
        # Act down
        new_game = game.move(GameStatus.DOWN)
        self.assertEqual(Point.ints_to_points(snake_down), new_game.snake)
        # Act up
        new_game = game.move(GameStatus.UP)
        self.assertEqual(Point.ints_to_points(snake_up), new_game.snake)

    def test_move_take_apple(self):
        # Arrange
        size = 5
        snake_start = [[3, 2], [2, 2], [1, 2]]
        snake_right = [[4, 2], [3, 2], [2, 2], [1, 2]]
        apple = [4, 2]
        game = GameStatus(size, snake_start, apple)
        # Act right and eat the apple
        new_game = game.move(GameStatus.RIGHT)
        self.assertEqual(Point.ints_to_points(snake_right), new_game.snake)
        self.assertNotEqual(game.snake, new_game.snake)

    def test_is_valid_input(self):
        size = 2
        snake_start = [[13, 2], [2, 2], [1, 2]]
        apple = [4, 2]

        # Size < 3
        self.assertFalse(GameStatus(size, snake_start, apple).is_valid_game())

        size = 10
        # Snake out of boundaries
        self.assertFalse(GameStatus(size, snake_start, apple).is_valid_game())

        # Apple out of boundaries
        size = 50
        apple = [51, 2]
        self.assertFalse(GameStatus(size, snake_start, apple).is_valid_game())

    def test_generate_new_apple(self):
        size = 10
        snake_start = [[5, 2], [2, 2], [1, 2]]
        apple = [4, 2]
        game = GameStatus(size, snake_start, apple)
        for i in range(1000):
            new_apple = game.generate_new_apple()
            self.assertFalse(new_apple in game.snake, "The apple: {} is in the snake {}".format(new_apple, game.snake))

    def test_clone(self):
        size = 10
        snake_start = [[3, 1], [2, 1], [1, 1]]
        apple = [4, 2]
        game = GameStatus(size, snake_start, apple)
        new_game = game.move(Point(0, 1))
        self.assertNotEqual(new_game.snake, game.snake)
        # We have not picked the apple so it stays in the same position
        self.assertEqual(new_game.apple, game.apple)
        new_game = new_game.move(Point(1, 0))
        # We have picked the apple so now it is different
        self.assertNotEqual(new_game.apple, game.apple)
