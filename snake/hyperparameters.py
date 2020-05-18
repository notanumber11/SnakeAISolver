import json
from collections import namedtuple


class HyperParameters:

    def __init__(self):
        self.iterations = None
        self.population_size = None
        self.mutation_rate = None
        self.games_to_play = None
        self.selection_threshold = None
        self.game_size = None
        self.snake_size = None
        self.mut_type = {
            "uniform": None,
            "gaussian": None,
        }
        self.cross_type = {
            "random": None,
            "single_point_binary": None,
            "simulated_binary": None
        }

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
            hyperparameters = HyperParameters()
            hyperparameters.__dict__ = data
        hyperparameters.self_validate()

    def self_validate(self):
        h = HyperParameters()
        all_fields_in_class = [f for f in dir(h) if not f.startswith('__') and not callable(getattr(h, f))]
        none_values = [f for f in all_fields_in_class if self.__dict__.get(f) is None]
        for f in all_fields_in_class:
            d = self.__dict__.get(f)
            full_d = h.__dict__.get(f)
            if type(d) == dict:
                none_values += ["{}[{}]".format(f,k) for k in full_d.keys() if d.get(k) is None]
        if none_values:
            raise ValueError("Some properties of hyperparameters are None" + str(none_values))


if __name__ == '__main__':
    hyperparameters = HyperParameters.load("hyperparameters_2.json")
