import abc
from typing import List

from game.game_status import GameStatus


class Solver(metaclass=abc.ABCMeta):

    def finished(self):
        print("{} finished".format(self))

    def __str__(self):
        val = str(type(self))
        s = val.rfind(".")
        e = val.rfind(">")
        return val[s+1:e-1]

    @abc.abstractmethod
    def solve(self, game_seed: GameStatus) -> List[GameStatus]:
        pass