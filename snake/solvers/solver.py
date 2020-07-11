import abc
from typing import List

from game.game_status import GameStatus


class Solver(metaclass=abc.ABCMeta):

    def __init__(self):
        self.movements_left = None

    def is_loop(self, prev: GameStatus, new: GameStatus):
        self.movements_left = prev.get_movements_left() if self.movements_left is None else self.movements_left
        self.movements_left -= 1
        if prev.apple != new.apple:
            self.movements_left = new.get_movements_left()
        if self.movements_left <= 0:
            print("Loop has been found !")
            return True
        return False

    def finished(self):
        print("{} finished".format(self))

    def __str__(self):
        """
        if type is <class 'solvers.genetic.genetic_algorithm.random_solver'> it
        returns random_solver.
        :return: instance type
        """
        val = str(type(self))
        s = val.rfind(".")
        e = val.rfind(">")
        return val[s + 1:e - 1]

    @abc.abstractmethod
    def solve(self, game_seed: GameStatus) -> List[GameStatus]:
        pass
