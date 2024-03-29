import random
from typing import List

from game.game_status import GameStatus
from game.point import Point
from solvers.solver import Solver


class DFSSolver(Solver):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.total_counter = 0
        self.dirs = {}

    def solve(self, game_seed: GameStatus) -> List[GameStatus]:
        game_statuses = []
        games = []
        self.counter = 0
        self.total_counter = 0
        current_game_status = game_seed
        while self.dfs(current_game_status, games, []):
            # print("It needed {} dfs to find an apple".format(self.counter))
            self.total_counter += self.counter
            self.counter = 0
            # Add all games except the first one since the dfs starts with the last game of the previous dfs
            game_statuses += games[1:]
            current_game_status = game_statuses[-1]
            games = []
        self.finished()
        print("{} dfs calls required to finish the game.".format(self.total_counter))
        return game_statuses

    def dfs(self, game: GameStatus, games: list, visited: list):
        self.counter += 1
        visited.append(game.head)
        games.append(game)
        for dir_ in random.sample(GameStatus.DIRS, len(GameStatus.DIRS)):
            # Is a valid direction move
            new_pos = Point(game.head.x + dir_.x, game.head.y + dir_.y)
            if new_pos not in visited and game.can_move_to_pos(new_pos):
                new_game = game.move(dir_)
                # If the new pos is the end
                if new_pos == game.apple:
                    games.append(new_game)
                    return True
                if self.dfs(new_game, games, visited):
                    return True
        visited.pop()
        games.pop()
        return False

    def __str__(self):
        return "DFSSolver"
