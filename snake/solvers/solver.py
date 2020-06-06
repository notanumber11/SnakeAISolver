import abc
from typing import List

from game.game_status import GameStatus


class Solver(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def solve(self, game_seed: GameStatus) -> List[GameStatus]:
        pass