from typing import List

from game.game_status import GameStatus
from game.point import Point


class HamiltonSolver:
    def solve(self, game_status: GameStatus) -> List[GameStatus]:
        game_statuses = []
        is_hamilton_path_found = self.hamilton(game_status, game_statuses, [], game_status.head)
        if not is_hamilton_path_found:
            print("Hamilton could not find a valid path")
            return [game_status]
        return self.apply_hamilton_path_until_finish_game(game_statuses, 0)

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

    def hamilton(self, game_status: GameStatus, game_statuses: List[GameStatus], visited: List[GameStatus], goal: Point):
        visited.append(game_status.head)
        game_statuses.append(game_status)

        for _dir in GameStatus.DIRS:
            new_pos = Point(game_status.head.x + _dir.x, game_status.head.y + _dir.y)
            # If the new pos is the end
            if new_pos == goal and len(visited) == game_status.size * game_status.size:
                return True
            if new_pos not in visited and game_status.can_move_to_pos(new_pos):
                new_game_status = game_status.move(_dir)
                if self.hamilton(new_game_status, game_statuses, visited, goal):
                    return True
        visited.pop()
        game_statuses.pop()
        return False

    def __str__(self):
        return "HamiltonSolver"
