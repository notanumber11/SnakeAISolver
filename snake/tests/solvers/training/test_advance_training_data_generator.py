import unittest

from game.game_status import GameStatus
from game.point import Point
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator as aT


class TestAdvanceTrainingDataGenerator(unittest.TestCase):
    
    def setUp(self):
        self.advance_training = aT()

    def test_vision_horizontal_snake(self):
        size = 5
        snake = [[x, 0] for x in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        true_dirs = [aT.RIGHT]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_vertical_snake(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        true_dirs = [aT.DOWN]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_diagonal(self):
        size = 5
        snake = [[col, 0] for col in range(size)]
        [snake.append([size - 1, row]) for row in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        print(game_status)
        true_dirs = [aT.RIGHT, aT.DOWN_RIGHT]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_all_directions_see_snake(self):
        size = 5
        snake = [[col, 0] for col in range(size)]
        [snake.append([size - 1, row]) for row in range(size)]
        [snake.append([col, size - 1]) for col in range(size - 1, -1, -1)]
        snake.append([0, size - 2])
        snake.append([1, size - 2])
        snake.reverse()
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        false_dirs = [aT.UP_LEFT]
        true_dirs = [_dir for _dir in aT.VISION if _dir not in false_dirs]
        self.assertTrue(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_get_wall_distance(self):
        size = 5
        snake = [[2, 2], [3, 2]]
        game_status = GameStatus(size, snake)
        for _dir in self.advance_training.VISION:
            self.assertEqual(2, self.advance_training._get_wall_distance(game_status, _dir))
        for _dir in self.advance_training.VISION:
            self.assertEqual(0.5, self.advance_training._get_size_normalize_wall_distance(game_status, _dir))

    def test_get_tail_dir_horizontal(self):
        size = 5
        snake = [[x, 0] for x in range(size)]
        game_status = GameStatus(size, snake)
        tail_dir = self.advance_training._get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.LEFT])
        game_status.snake.reverse()
        tail_dir = self.advance_training._get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.RIGHT])

    def test_get_tail_dir_vertically(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        tail_dir = self.advance_training._get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.UP])
        game_status.snake.reverse()
        tail_dir = self.advance_training._get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.DOWN])

    def test_get_apple_vision(self):
        size = 5
        snake = [[0, x] for x in range(size - 1)]
        apple = [0, size - 1]
        game_status = GameStatus(size, snake, apple)
        true_dirs = [aT.DOWN, ]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((self.advance_training._get_apple_vision(game_status, _dir) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_apple_vision(game_status, _dir) for _dir in false_dirs)))

    def test_get_input_from_game_status(self):
        size = 5
        snake = [[1, x] for x in range(size - 1)]
        apple = [0, 1]
        game_status = GameStatus(size, snake, apple)
        print(game_status)
        input = self.advance_training.get_input_from_game_status(game_status)
        # The game looks as follows:
        """[[ 0    H    0    0    0 ]
            [ X    1    0    0    0 ]
            [ 0    1    0    0    0 ]
            [ 0    1    0    0    0 ]
            [ 0    0    0    0    0 ]]"""
        expected_input = [0.0, 0, 0,   # UP
                          0.0, 0, 0,   # UP_RIGHT
                          0.75, 0, 0,  # RIGHT
                          0.75, 0, 0,  # DOWN_RIGHT
                          1.0, 0, 1,   # DOWN
                          0.25, 1, 0,  # DOWN_LEFT
                          0.25, 0, 0,  # LEFT
                          0.0, 0, 0,   # UP_LEFT
                          1, 0, 0, 0]  # Snake tail dir
        self.assertEqual(28, len(input))
        self.assertTrue(all(0 <= x <= 1 for x in input))
        self.assertEqual(input, expected_input)
