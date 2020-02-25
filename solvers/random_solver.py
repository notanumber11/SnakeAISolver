import random
from typing import List

from model.game_status import GameStatus
from model.point import Point


class RandomSolver:
    def __init__(self):
        pass

    def solve(self, game_seed: GameStatus)-> List[GameStatus]:
        game_status = game_seed
        result = [game_status]
        prev_dir = Point(game_status.head.x - game_status.snake[1].x, game_status.head.y - game_status.snake[1].y)
        while game_status.is_valid_game():
            dirs = random.sample(GameStatus.DIRS, len(GameStatus.DIRS))
            for dir_ in dirs:
                # Do not allow a random movement into the opposite direction
                if dir_ == self.get_opposite_dir(prev_dir):
                    continue
                game_status = game_status.move(dir_)
                result.append(game_status)
                prev_dir = dir_
                break
        return result

    def get_opposite_dir(self, dir_: Point):
        x = 0
        if dir_.x != 0:
            x = 1 if dir_.x == -1 else -1
        y = 0
        if dir_.y != 0:
            y = 1 if dir_.y == -1 else -1
        return Point(x, y)