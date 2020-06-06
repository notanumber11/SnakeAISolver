import unittest

from game.game_status import GameStatus
from solvers.data_providers.advance_training_data_generator import AdvanceTrainingDataGenerator as aT


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
        self.assertTrue(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_vertical_snake(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        true_dirs = [aT.DOWN]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_vision_diagonal(self):
        size = 5
        snake = [[col, 0] for col in range(size)]
        [snake.append([size - 1, row]) for row in range(size)]
        game_status = GameStatus(size, snake)
        snake_set = set(game_status.snake)
        print(game_status)
        true_dirs = [aT.RIGHT, aT.DOWN_RIGHT]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

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
        self.assertTrue(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in true_dirs)))
        self.assertFalse(
            all((self.advance_training._get_body_vision(game_status, _dir, snake_set) for _dir in false_dirs)))

    def test_get_wall_distance(self):
        size = 5
        snake = [[2, 2], [3, 2]]
        game_status = GameStatus(size, snake)
        for _dir in self.advance_training.VISION:
            distance_to_wall = self.advance_training._get_wall_distance(game_status, _dir)
            self.assertEqual(2, distance_to_wall)
        for _dir in self.advance_training.VISION:
            distance_to_wall = self.advance_training._get_wall_distance(game_status, _dir)
            self.assertEqual(0.5, self.advance_training._get_normalize_distance(game_status, distance_to_wall))

    def test_get_tail_dir_horizontal(self):
        size = 5
        snake = [[x, 0] for x in range(size)]
        game_status = GameStatus(size, snake)
        tail_dir = self.advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.LEFT])
        game_status.snake.reverse()
        tail_dir = self.advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.RIGHT])

    def test_get_tail_dir_vertically(self):
        size = 5
        snake = [[0, x] for x in range(size)]
        game_status = GameStatus(size, snake)
        tail_dir = self.advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.UP])
        game_status.snake.reverse()
        tail_dir = self.advance_training.get_tail_dir(game_status)
        self.assertEqual(tail_dir, aT.DIR_TO_VECTOR[aT.DOWN])

    def test_get_apple_vision(self):
        size = 5
        snake = [[0, x] for x in range(size - 1)]
        apple = [0, size - 1]
        game_status = GameStatus(size, snake, apple)
        true_dirs = [aT.DOWN]
        false_dirs = [_dir for _dir in aT.VISION if _dir not in true_dirs]
        self.assertTrue(all((self.advance_training._get_apple_vision(game_status, _dir) for _dir in true_dirs)))
        self.assertFalse(all((self.advance_training._get_apple_vision(game_status, _dir) for _dir in false_dirs)))

    def test_get_input_from_game_status(self):
        size = 5
        snake = [[1, x] for x in range(size - 1)]
        apple = [0, 1]
        game_status = GameStatus(size, snake, apple)
        input = self.advance_training.get_input_from_game_status(game_status)
        print(game_status)
        # The game looks as follows:
        """[[ 0    H    0    0    0 ]
            [ X    1    0    0    0 ]
            [ 0    1    0    0    0 ]
            [ 0    1    0    0    0 ]
            [ 0    0    0    0    0 ]]"""
        # Distance to wall, apple vision, body vision
        expected_input = [0.0, 0, 0,  # UP
                          0.0, 0, 0,  # UP_RIGHT
                          0.75, 0, 0,  # RIGHT
                          0.75, 0, 0,  # DOWN_RIGHT
                          1.0, 0, 1,  # DOWN
                          0.25, 1, 0,  # DOWN_LEFT
                          0.25, 0, 0,  # LEFT
                          0.0, 0, 0,  # UP_LEFT
                          1, 0, 0, 0]  # Snake tail dir
        self.assertEqual(28, len(input))
        self.assertTrue(all(0 <= x <= 1 for x in input))
        self.assertEqual(input, expected_input)

    def test_get_input_from_game_status_long_snake(self):
        size = 5
        snake = [[0, x] for x in range(size)]  # Vertical left
        snake += [[1, 4], [2, 4], [3, 4], [4, 4]]  # Horizontal bottom
        snake += [[4, 3], [4, 2], [4, 1], [4, 0]]  # Vertical right
        apple = [2, 1]
        game_status = GameStatus(size, snake, apple)
        input = self.advance_training.get_input_from_game_status(game_status)
        print(game_status)
        # Distance to wall, apple vision, body vision
        expected_input = [0.0, 0, 0,  # UP
                          0.0, 0, 0,  # UP_RIGHT
                          1.0, 0, 1,  # RIGHT
                          1.0, 0, 1,  # DOWN_RIGHT
                          1.0, 0, 1,  # DOWN
                          0.0, 0, 0,  # DOWN_LEFT
                          0.0, 0, 0,  # LEFT
                          0.0, 0, 0,  # UP_LEFT
                          0, 1, 0, 0]  # Snake tail dir
        self.assertEqual(28, len(input))
        self.assertTrue(all(0 <= x <= 1 for x in input))
        self.assertEqual(input, expected_input)

    def test_get_input_from_game_status_long_snake_circle(self):
        size = 5
        snake = [[1, x] for x in range(1, size)]  # Vertical left
        snake += [[2, 4], [2, 4], [3, 4], [4, 4]]  # Horizontal bottom
        snake += [[4, 3], [4, 2], [4, 1], [4, 0]]  # Vertical right
        snake += [[3, 0], [2, 0]]
        apple = [2, 1]
        game_status = GameStatus(size, snake, apple)
        input = self.advance_training.get_input_from_game_status(game_status)
        print(game_status)
        # Distance to wall, apple vision, body vision
        expected_input = [0.25, 0, 0,  # UP
                          0.25, 0, 1,  # UP_RIGHT
                          0.75, 1, 1,  # RIGHT
                          0.75, 0, 1,  # DOWN_RIGHT
                          0.75, 0, 1,  # DOWN
                          0.25, 0, 0,  # DOWN_LEFT
                          0.25, 0, 0,  # LEFT
                          0.25, 0, 0,  # UP_LEFT
                          0, 0, 0, 1]  # Snake tail dir
        self.assertEqual(28, len(input))
        self.assertTrue(all(0 <= x <= 1 for x in input))
        self.assertEqual(input, expected_input)
