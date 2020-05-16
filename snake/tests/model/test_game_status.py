import math
import unittest

from game import game_seed_creator
from game.game_status import GameStatus
from game.point import Point


class TestGameStatus(unittest.TestCase):

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

    def test_is_valid_game(self):
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
        # Can not move backwards when snake is size 2
        snake_start = [[2, 2], [1, 2]]
        game = GameStatus(size, snake_start)
        self.assertFalse(game.can_move_to_dir(GameStatus.LEFT))
        new_game = game.move(GameStatus.LEFT)
        self.assertFalse(new_game.is_valid_game())

    def test_generate_new_apple(self):
        size = 10
        snake_start = [[5, 2], [2, 2], [1, 2]]
        apple = [4, 2]
        game = GameStatus(size, snake_start, apple)
        for i in range(1000):
            new_apple = game._generate_new_apple()
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

    def test_angle(self):
        game_status = GameStatus(6, [[0, 0], [1, 0]])
        # x1>x2 , y1=y2 - Snake should move right
        apple = Point(4, 0)
        head = Point(-1, 0)
        self.assertEqual(0, math.degrees(game_status.get_angle(apple, head)))
        # x1<x2 , y1=y2 - Snake should move left
        apple = Point(-4, 0)
        head = Point(-1, 0)
        self.assertEqual(180, math.degrees(game_status.get_angle(apple, head)))
        # x1=x2, y1>y2 - Snake should move up
        apple = Point(0, 5)
        head = Point(0, 4)
        self.assertEqual(90, math.degrees(game_status.get_angle(apple, head)))
        # x1=x2, y1<y2 - Snake should move down
        apple = Point(0, -5)
        head = Point(0, 4)
        self.assertEqual(-90, math.degrees(game_status.get_angle(apple, head)))
        # x1>x2, y1>y2 -Snake should move up or right
        apple = Point(3, 3)
        head = Point(1, 1)
        self.assertEqual(45, math.degrees(game_status.get_angle(apple, head)))

    def test_can_not_move(self):
        # Arrange
        size = 5
        snake_start = [[0, 0], [1, 0]]
        apple = [4, 4]
        game = GameStatus(size, snake_start, apple)
        # Can not go backwards
        self.assertFalse(game.can_move_to_dir(GameStatus.RIGHT))
        # Crash with left wall
        self.assertFalse(game.can_move_to_dir(GameStatus.LEFT))
        # Crash with up wall
        self.assertFalse(game.can_move_to_dir(GameStatus.UP))
        # Can move down
        self.assertTrue(game.can_move_to_dir(GameStatus.DOWN))

    def test_can_not_move_backwards(self):
        # Arrange
        size = 5
        snake_start = [[2, 2], [2, 3]]
        game = GameStatus(size, snake_start)
        self.assertFalse(game.can_move_to_dir(GameStatus.DOWN))
        snake_start = [[2, 2], [2, 1]]
        game = GameStatus(size, snake_start)
        self.assertFalse(game.can_move_to_dir(GameStatus.UP))
        snake_start = [[2, 2], [3, 2]]
        game = GameStatus(size, snake_start)
        self.assertFalse(game.can_move_to_dir(GameStatus.RIGHT))
        snake_start = [[2, 2], [1, 2]]
        game = GameStatus(size, snake_start)
        self.assertFalse(game.can_move_to_dir(GameStatus.LEFT))

    def equality_list(self):
        a = [1, 3]
        b = [1, 3]
        c = [1, 4]
        self.assertEqual(a, b)
        self.assertNotEqual(b, c)

    def test_game_is_able_to_finish_successfully(self):
        size = 2
        snake_start = [[0, 0], [0, 1], [1, 1], [1,0]]
        game = GameStatus(size, snake_start)
        game.is_valid_game()
        self.assertTrue(game.is_full_finished())

    def test_get_number_of_holes(self):
        game_status = GameStatus(6, [[2,2], [1,2]])
        max_number_of_movements = game_status.get_number_of_holes()
        self.assertEqual(max_number_of_movements, 34)

    def test_print(self):
        snake = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
        apple = [0, 0]
        board_size = 6
        game_status = GameStatus(board_size, snake, apple)
        expected_str = """[[ X    0    0    0    0    0 ]
 [ 0    0    0    0    0    0 ]
 [ 0    H    1    1    1    1 ]
 [ 0    0    0    0    0    0 ]
 [ 0    0    0    0    0    0 ]
 [ 0    0    0    0    0    0 ]]"""
        self.assertEqual(expected_str, game_status.__str__())
