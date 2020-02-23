import random

from model.game import Game
from model.point import Point


class DFSSolver:

    def solve(self, game: Game):
        games_end = []
        games = []
        while self.dfs(game, games, []):
            # Add all games except the first one since the dfs starts with the last game of the previous dfs
            games_end += games[1:]
            game = games_end[-1]
            games = []
        return games_end

    def dfs(self, game: Game, games: list, visited: list):
        visited.append(game.head)
        games.append(game)

        dirs = random.sample(Game.DIRS, len(Game.DIRS))
        for dir in dirs:
            # Is a valid direction move
            new_pos = Point(game.head.x + dir.x, game.head.y + dir.y)
            if new_pos not in visited and game.can_move_to_pos(new_pos):
                new_game = game.move(dir)
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
