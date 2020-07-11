import solvers.distance_vision_genetic_solver
from solvers.training_data_generators import data_utils
from solvers.training_data_generators.classification.short_vision import ShortVision


class ShortVisionGeneticSolver(solvers.distance_vision_genetic_solver.DistanceVisionGeneticSolver):

    def __init__(self, path_model=None):
        super().__init__(path_model)
        best_path_model = r"models/basic_genetic/pop=1000_sel=0.1_mut_0.05_it_10000_games_1_game_size_6/76_____completion_36.0_36.0___1.00_____movements_327.0"
        self.path_model = path_model if path_model is not None else best_path_model
        self.data_provider = ShortVision()
        self.model = data_utils.load_model(self.path_model)
