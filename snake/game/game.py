from typing import List

from game.game_status import GameStatus


class Game:
    def __init__(self, game_statuses: List[GameStatus], loop=False):
        self.solved = False
        self.game_statuses = game_statuses
        self.was_stack_in_loop = loop
        self._is_finished = False

    def generate_report(self):
        if not self.solved:
            raise ValueError("The game needs to be solved before generating a report...")
        pass

    def next(self) -> GameStatus:
        if len(self.game_statuses) > 1:
            return self.game_statuses.pop(0)
        self._is_finished = True
        return self.game_statuses[0]

    def is_finished(self):
        return self._is_finished
