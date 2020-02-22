from model.game import Game


class BasicSolver:

    def solve(self, game: Game):
        games = [game]
        while game.is_valid_game():
            game = game.move(self.next_dir(game))
            games.append(game)
        return games

    def next_dir(self, game):
        return Game.LEFT
