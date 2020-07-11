import random

from game.game_status import GameStatus
from solvers.solver import Solver


class SurvivalRandomSolver(Solver):

    def solve(self, prev: GameStatus):
        games = [prev]
        while prev.is_valid_game():
            new = prev.move(self.get_best_movement(prev))
            games.append(prev)
            prev = new
        self.finished()
        return games

    def get_best_movement(self, game_status: GameStatus):
        dirs = random.sample(game_status.DIRS, len(GameStatus.DIRS))
        for dir in dirs:
            if game_status.can_move_to_dir(dir):
                return dir
        return dirs[0]
