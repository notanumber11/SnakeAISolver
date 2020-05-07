import unittest

from game.game_status import GameStatus
from game.point import Point
from solvers.training.advance_training_data_generator import AdvanceTrainingDataGenerator as aT


class TestAdvanceTrainingDataGenerator(unittest.TestCase):

    def test_vision_horizontal_snake(self):
        size = 5
        snake = [[x, 0] for x in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        advance_training = aT()
        true_dirs = [aT.RIGHT]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_vertical_snake(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        advance_training = aT()
        true_dirs = [aT.DOWN]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_diagonal(self):
        size = 5
        snake = [[col, 0] for col in range(size)]
        [snake.append([size-1, row]) for row in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        advance_training = aT()
        print(game_status)
        true_dirs = [aT.RIGHT, aT.DOWN_RIGHT]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_all_directions_see_snake(self):
        size = 5
        snake = [[col, 0] for col in range(size)]
        [snake.append([size-1, row]) for row in range(size)]
        [snake.append([col, size-1]) for col in range(size-1, -1, -1)]
        snake.append([0, size-2])
        snake.append([1, size-2])
        snake.reverse()
        game_status = GameStatus(size, snake)
        advance_training = aT()
        snake_set = set(game_status.snake)
        false_dirs = [aT.UP_LEFT]
        true_dirs = [_dir for _dir in aT.VISION if _dir not in false_dirs]
        self.assertTrue(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(all((advance_training.get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_get_wall_distance(self):
        size = 5
        snake = [[2,2], [3,2]]
        game_status = GameStatus(size, snake)
        advance_training = aT()
        for _dir in advance_training.VISION:
            self.assertEqual(2, advance_training.get_wall_distance(game_status, _dir))

    def test_get_tail_dir_horizontal(self):
        size = 5
        snake = [[x, 0] for x in range(size)]
        game_status = GameStatus(size, snake)
        advance_training = aT()
        tail_dir = advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.LEFT])
        game_status.snake.reverse()
        tail_dir = advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.RIGHT])

    def test_get_taiql_dir_vertically(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        advance_training = aT()
        tail_dir = advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.UP])
        game_status.snake.reverse()
        tail_dir = advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.DOWN])