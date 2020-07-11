import abc
from typing import List

from game.game_status import GameStatus


class TrainingDataGenerator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_input_from_game_status(self, game_status: GameStatus) -> List[List[float]]:
        pass
