import random

from model.game_status import GameStatus


class BasicSolver:

    def solve(self, game: GameStatus):
        games = [game]
        while game.is_valid_game():
            game = game.move(self.next_dir(game))
            games.append(game)
        return games

    def next_dir(self, game_status: GameStatus):
        dirs = random.sample(game_status.DIRS, len(GameStatus.DIRS))
        for dir in dirs:
            if game_status.can_move_to_dir(dir):
                return dir
        return dirs[0]

    def __str__(self):
        return "BasicSolver"
