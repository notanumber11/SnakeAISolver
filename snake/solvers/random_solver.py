import random
from typing import List

from game.game_status import GameStatus
from game.point import Point
from solvers.solver import Solver


class RandomSolver(Solver):

    def solve(self, game_seed: GameStatus) -> List[GameStatus]:
        game_status = game_seed
        result = [game_status]
        prev_dir = game_status.prev_dir
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
        self.finished()
        return result

    def get_opposite_dir(self, dir_: Point):
        x = 0
        if dir_.x != 0:
            x = 1 if dir_.x == -1 else -1
        y = 0
        if dir_.y != 0:
            y = 1 if dir_.y == -1 else -1
        return Point(x, y)
