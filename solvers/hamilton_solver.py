from model.game import Game
from model.point import Point


class HamiltonSolver:
    def solve(self, game: Game):
        games = []
        self.hamilton(game, games, [], game.head)
        return self.next_game(games, 0)

    def next_game(self, games, pos):
        last = games[-1]
        holes = last.size * last.size - len(last.snake)
        while holes > 0:
            ref = games[pos]
            dir = Point(ref.head.x - last.head.x, ref.head.y - last.head.y)
            last = last.move(dir)
            holes = last.size * last.size - len(last.snake)
            games.append(last)
            pos += 1
        return games

    def hamilton(self, game: Game, games: list, visited: list, goal: Point):
        visited.append(game.head)
        games.append(game)

        for dir in Game.DIRS:
            # Is a valid direction move
            new_pos = Point(game.head.x + dir.x, game.head.y + dir.y)
            # If the new pos is the end
            if new_pos == goal and len(visited) == game.size * game.size:
                return True
            if new_pos not in visited and game.can_move_to_pos(new_pos):
                new_game = game.move(dir)
                if self.hamilton(new_game, games, visited, goal):
                    return True
        visited.pop()
        games.pop()
        return False

    def __str__(self):
        return "HamiltonSolver"
