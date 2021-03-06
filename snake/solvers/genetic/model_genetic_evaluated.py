from typing import List

import numpy as np

from game.game import Game


class ModelGeneticEvaluated:
    def __init__(self, games: List[Game], model_genetic):
        self.model_genetic = model_genetic
        self.snake_length = 0
        self.movements = 0
        self.size = 0
        self.apples = 0
        for game in games:
            if game.was_stack_in_loop:
                self.snake_length = 0
                self.apples = 0
                self.movements = 1
                self.size = game.game_statuses[-1].size
                return
            else:
                self.apples += (len(game.game_statuses[-1].snake) - len(game.game_statuses[0].snake))
                self.snake_length += len(game.game_statuses[-1].snake)
                self.movements += len(game.game_statuses) - 1
                self.size += game.game_statuses[-1].size
        self.snake_length /= len(games)
        self.movements /= len(games)
        self.size /= len(games)
        self.apples /= len(games)

    def _equal_model_genetic(self, a, b):
        return all(np.array_equal(a[i], b[i]) for i in range(len(a)))

    def fitness(self):
        movements = self.movements / self.size ** 2 * 100
        apples = self.apples / self.size ** 2 * 100
        fitness = movements + apples ** 2 / movements + apples ** 2.5 - (0.01 * movements) ** 2
        return fitness

    def basic_fitness(self):
        fitness = self.snake_length ** 3 - self.movements
        if fitness < 0:
            fitness = 0
        return fitness

    def other_fitness(self):
        fitness = self.movements + (2 ** self.apples + 500 * self.apples ** 2.1) - (
                0.25 * self.movements ** 1.3 * self.apples ** 1.2)
        return fitness

    def __str__(self):
        return "        Apples={:.2f} - Size={:.2f} - Snake length={:.2f} - Movements={:.2f} - Fitness={:.2f}" \
            .format(self.apples, self.size, self.snake_length, self.movements, self.fitness())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None:
            return False
        if self.snake_length != other.snake_length or self.movements != other.movements \
                or self.size != other.size or self.apples != other.apples:
            return False
        if not self._equal_model_genetic(self.model_genetic, other.model_genetic):
            return False
        return True

    def __hash__(self):
        result = 0
        for l in self.model_genetic:
            result += np.sum(l)
        return int(self.snake_length * self.movements + self.size + result * self.apples)

    def summary(self) -> str:
        return "apples_{:.1f}_size_{:.1f}_movements_{:.1f}" \
            .format(self.apples,
                    self.size ** 2,
                    self.movements)
