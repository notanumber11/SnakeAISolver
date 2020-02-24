from model.game_status import GameStatus


class BasicSolver:

    def solve(self, game: GameStatus):
        games = [game]
        while game.is_valid_game():
            game = game.move(self.next_dir(game))
            games.append(game)
        return games

    def next_dir(self, game_status: GameStatus):
        for dir in game_status.DIRS:
            if game_status.is_valid_dir(dir):
                return dir
        return GameStatus.LEFT

    def __str__(self):
        return "BasicSolver"
