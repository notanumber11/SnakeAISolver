from typing import List

from model.game_status import GameStatus
from model.point import Point


class HamiltonSolver:
    def solve(self, game_status: GameStatus) -> List[GameStatus]:
        games = []
        is_hamilton_path_found = self.hamilton(game_status, games, [], game_status.head)
        if not is_hamilton_path_found:
            print("Hamilton could not find a valid path")
            return [game_status]
        return self.apply_hamilton_path_until_finish_game(games, 0)

    def apply_hamilton_path_until_finish_game(self, statuses: List[GameStatus], pos: int) -> List[GameStatus]:
        last_status = statuses[-1]
        holes = last_status.size * last_status.size - len(last_status.snake)
        while holes > 0:
            ref = statuses[pos]
            dir = Point(ref.head.x - last_status.head.x, ref.head.y - last_status.head.y)
            last_status = last_status.move(dir)
            holes = last_status.size * last_status.size - len(last_status.snake)
            statuses.append(last_status)
            pos += 1
        return statuses

    def hamilton(self, game_status: GameStatus, games: list, visited: list, goal: Point):
        visited.append(game_status.head)
        games.append(game_status)

        for dir in GameStatus.DIRS:
            # Is a valid direction move
            new_pos = Point(game_status.head.x + dir.x, game_status.head.y + dir.y)
            # If the new pos is the end
            if new_pos == goal and len(visited) == game_status.size * game_status.size:
                return True
            if new_pos not in visited and game_status.can_move_to_pos(new_pos):
                new_game = game_status.move(dir)
                if self.hamilton(new_game, games, visited, goal):
                    return True
        visited.pop()
        games.pop()
        return False

    def __str__(self):
        return "HamiltonSolver"
