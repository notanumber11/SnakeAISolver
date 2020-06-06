from typing import List

import numpy as np

from game.game_status import GameStatus
from game.point import Point
from solvers.data_providers.binary_vision_training_data_generator import BinaryVisionTrainingDataGenerator


class DistanceVisionTrainingDataGenerator:
    VISION = BinaryVisionTrainingDataGenerator.VISION

    def __init__(self):
        self.ag = BinaryVisionTrainingDataGenerator()
        pass

    def get_input_from_game_status(self, game_status: GameStatus) -> List[float]:
        training_data = []
        snake_set = set(game_status.snake)

        for i in range(len(self.VISION)):
            distances = self._get_distances(game_status, self.VISION[i], snake_set)
            training_data += distances

        tail_dir = self.ag.get_tail_dir(game_status)
        training_data += tail_dir
        return training_data

    def _get_distances(self, game_status: GameStatus, _dir: Point, snake_set) -> List[float]:
        """
        :param _dir:
        :param snake_set: set with snake points
        :return: distance to wall, distance to apple and distance to snake
            The distance follows the following pattern: d = 1/number of cells distance
                distance infinite -> 0
                distance 100 -> 0.01
                distance 10 -> 0.1
                distance 2 -> 0.5
                distance 1 -> 1
        """
        new_pos = game_status.snake[0] + _dir
        distance = 1
        apple_distance = np.inf
        apple_found = False
        snake_distance = np.inf
        snake_found = False
        apple = game_status.apple
        while game_status.is_inside_board(new_pos):
            # Apple distance
            if new_pos == apple and not apple_found:
                apple_distance = distance
                apple_found = True
            # Snake distance
            if new_pos in snake_set and not snake_found:
                snake_distance = distance
                snake_found = True
            # Wall distance
            distance += 1
            # Advance loop
            new_pos = new_pos + _dir

        wall_distance = round(1.0 / distance, 2)
        apple_distance = round(1.0 / apple_distance, 2)
        snake_distance = round(1.0 / snake_distance, 2)
        return [wall_distance, apple_distance, snake_distance]
