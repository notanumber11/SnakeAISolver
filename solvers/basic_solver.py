from model.game_status import GameStatus


class BasicSolver:

    def solve(self, game: GameStatus):
        games = [game]
        while game.is_valid_game():
            game = game.move(self.next_dir(game))
            games.append(game)
        return games

    def next_dir(self, game):
        return GameStatus.LEFT

    def __str__(self):
        return "BasicSolver"