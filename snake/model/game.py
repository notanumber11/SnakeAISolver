from typing import List

from model.game_status import GameStatus


class Game:
    def __init__(self, game_statuses: List[GameStatus]):
        self.solved = False
        self.game_statuses = game_statuses

    def generate_report(self):
        if not self.solved:
            raise ValueError("The game needs to be solved before generating a report...")
        pass

    def next(self) -> GameStatus:
        if len(self.game_statuses) > 1:
            return self.game_statuses.pop(0)
        return self.game_statuses[0]
