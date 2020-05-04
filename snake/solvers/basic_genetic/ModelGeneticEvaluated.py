import numpy as np

class ModelGeneticEvaluated:
    def __init__(self, snake_length, movements, reward, size, model_genetic):
        self.snake_length = snake_length
        self.movements = movements
        self.reward = reward
        self.size = size
        self.model_genetic = model_genetic

    def fitness(self):
        fitness = self.snake_length**3 - self.movements
        if fitness < 0:
            fitness = 0
        return fitness

    def __str__(self):
        return "        Fitness={:.2f} - Snake length={:.2f} - Movements={:.2f} - Reward={:.2f}" \
            .format(self.fitness(), self.snake_length, self.movements, self.reward)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None:
            return False
        if self.snake_length != other.snake_length or self.movements != other.movements \
                or self.reward != other.reward or self.size != other.size:
            return False
        if not self._equal_model_genetic(self.model_genetic, other.model_genetic):
            return False
        return True

    def __hash__(self):
        result = 0
        for l in self.model_genetic:
            result += np.sum(l)
        return int(self.snake_length * self.movements + self.reward * self.size + result)

    def _equal_model_genetic(self, a, b):
        return all(np.array_equal(a[i], b[i]) for i in range(len(a)))