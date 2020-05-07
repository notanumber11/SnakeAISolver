from typing import List

from game.game_status import GameStatus
from game.point import Point


class AdvanceTrainingDataGenerator:
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
        DOWN:  [0, 1, 0, 0],
        LEFT: [0, 0, 1, 0],
        RIGHT: [0, 0, 0, 1]
    }

    # 8 vision lines
    # distance to wall
    # is there an apple?*
    # is there part of the snake?*
    # * can be either yes/no/distance
    # 8 directions with 3 variables each:
    #   a) distance to wall
    #   b) is snake there boolean
    #   c) is apple there boolean
    def __init__(self):
        pass

    def get_input_from_game_status(game_status: GameStatus):
        pass

    def get_body_vision(self, game_status: GameStatus, _dir: Point, snake_set) -> bool:
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

    def get_wall_distance(self, game_status: GameStatus, _dir: Point) -> int:
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
        tail_dir = game_status.snake[len_s-2] - game_status.snake[len_s-1]
        return AdvanceTrainingDataGenerator.DIR_TO_VECTOR[tail_dir]
