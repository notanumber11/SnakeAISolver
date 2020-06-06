from typing import List

from game.game_status import GameStatus
from game.point import Point


class BinaryVisionTrainingDataGenerator:
    """
    To understand the data generated please check AdvanceTrainingDataGeneratorTest
    """
    LABELS = ["up", "down", "left", "right",
              "angle to apple", "reward"]

    UP = Point(0, -1)
    UP_RIGHT = Point(1, -1)
    RIGHT = Point(1, 0)
    DOWN_RIGHT = Point(1, 1)
    DOWN = Point(0, 1)
    DOWN_LEFT = Point(-1, 1)
    LEFT = Point(-1, 0)
    UP_LEFT = Point(-1, -1)
    VISION = [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP]

    DIR_TO_VECTOR = {
        UP: [1, 0, 0, 0],
        DOWN: [0, 1, 0, 0],
        LEFT: [0, 0, 1, 0],
        RIGHT: [0, 0, 0, 1]
    }

    def get_input_from_game_status(self, game_status: GameStatus) -> List[float]:
        """
        The goal of this method is to create the following input data:
        A vector of dimension 28 that includes 24 inputs for vision and  4 inputs for the direction of the tail of the snake

        The 24 inputs for vision are looking at the 8 directions defined by VISION list.
        Each element on VISION list will have 3 values:
            - Distance to the wall
            - Is there an apple (boolean)
            - Is there snake body (boolean)
        An example for one direction will be: [5, 0, 1] meaning that there is a distance of 5 to the wall, the apple is not
        in that direction and the body of the snake is.

        The 4 inputs for tail direction are a vector that indicates if the tail direction with the following boolean values:
        [IS_GOING_UP, IS_GOING_DOWN, IS_GOING_LEFT, IS_GOING_RIGHT].
        An example for the tail going down will be: [0, 1, 0, 0].
        """
        training_data = []
        snake_set = set(game_status.snake)

        for i in range(len(self.VISION)):
            distance_to_wall = self._get_wall_distance(game_status, self.VISION[i])
            wall_distance = self._get_normalize_distance(game_status, distance_to_wall)
            training_data.append(wall_distance)
            apple_vision = int(self._get_apple_vision(game_status, self.VISION[i]))
            training_data.append(apple_vision)
            body_vision = int(self._get_body_vision(game_status, self.VISION[i], snake_set))
            training_data.append(body_vision)

        tail_dir = self.get_tail_dir(game_status)
        training_data += tail_dir
        return training_data

    def _get_normalize_distance(self, game_status: GameStatus, distance: int):
        return round(distance / (game_status.size - 1), 2)

    def _get_body_vision(self, game_status: GameStatus, _dir: Point, snake_set) -> bool:
        """
        :param _dir:
        :param snake_set: set with snake points
        :return: True if the snake can see its own body in that direction
        """
        new_pos = game_status.snake[0] + _dir
        while game_status.is_inside_board(new_pos):
            if new_pos in snake_set:
                return True
            new_pos = new_pos + _dir
        return False

    def _get_wall_distance(self, game_status: GameStatus, _dir: Point) -> int:
        """
        :param _dir:
        :param snake_set: set with snake points
        :return: True if the snake can see its own body in that direction
        """
        new_pos = game_status.snake[0] + _dir
        distance = 0
        while game_status.is_inside_board(new_pos):
            new_pos = new_pos + _dir
            distance += 1
        return distance

    def get_tail_dir(self, game_status: GameStatus):
        len_s = len(game_status.snake)
        tail_dir = game_status.snake[len_s - 2] - game_status.snake[len_s - 1]
        return BinaryVisionTrainingDataGenerator.DIR_TO_VECTOR[tail_dir]

    def _get_apple_vision(self, game_status: GameStatus, _dir: Point):
        new_pos = game_status.snake[0] + _dir
        while game_status.is_inside_board(new_pos):
            if new_pos == game_status.apple:
                return True
            new_pos = new_pos + _dir
        return False
