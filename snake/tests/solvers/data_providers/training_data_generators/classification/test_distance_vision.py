import unittest

from game.game_status import GameStatus
from solvers.training_data_generators.classification.distance_vision import DistanceVision


class TestBinaryVisionTrainingDataGenerator(unittest.TestCase):

    def setUp(self):
        self.advance_training = DistanceVision()

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
        expected_input = [0.5, 0.0, 0.0,  # UP
                          0.5, 0.0, 1,  # UP_RIGHT
                          0.25, 1, 0.33,  # RIGHT
                          0.25, 0.0, 0.33,  # DOWN_RIGHT
                          0.25, 0.0, 1,  # DOWN
                          0.5, 0.0, 0.0,  # DOWN_LEFT
                          0.5, 0.0, 0.0,  # LEFT
                          0.5, 0.0, 0.0,  # UP_LEFT
                          0, 0, 0, 1]  # Snake tail dir
        self.assertEqual(28, len(input))
        self.assertTrue(all(0 <= x <= 1 for x in input))
        self.assertEqual(expected_input, input)
