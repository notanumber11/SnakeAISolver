from typing import List

from game.game_status import GameStatus
import solvers.training_data_generators.regression.short_vision_regression
from solvers.training_data_generators.training_data_generator import TrainingDataGenerator


class ShortVision(TrainingDataGenerator):

    def __init__(self):
        self._short_vision_regression = solvers.training_data_generators.regression.short_vision_regression.ShortVisionRegression()

    def get_input_from_game_status(self, game_status: GameStatus) -> List[List[float]]:
        input = self._short_vision_regression.get_input_from_game_status(game_status)[0]
        input = input[4:]
        return [input]
