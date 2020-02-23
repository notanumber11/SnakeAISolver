import random

from model.game_status import GameStatus
from model.point import Point


class DFSSolver:

    def solve(self, game: GameStatus):
        games_end = []
        games = []
        self.counter = 0
        self.total_counter = 0
        while self.dfs(game, games, []):
            # print("It needed {} dfs to find an apple".format(self.counter))
            self.total_counter += self.counter
            self.counter = 0
            # Add all games except the first one since the dfs starts with the last game of the previous dfs
            games_end += games[1:]
            game = games_end[-1]
            games = []
        print("It needed {} dfs to finish the game".format(self.total_counter))
        return games_end

    def dfs(self, game: GameStatus, games: list, visited: list):
        self.counter += 1
        visited.append(game.head)
        games.append(game)

        dirs = random.sample(GameStatus.DIRS, len(GameStatus.DIRS))
        # dirs = GameStatus.DIRS
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
