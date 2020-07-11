from random import seed
from typing import List

import game.game_seed_creator
import solvers.distance_vision_genetic_solver as ag
import solvers.reward_based_dnn_solver
import solvers.short_vision_genetic_solver
import solvers.survival_random_solver
import utils.timing
from game.game import Game
from solvers.dfs_solver import DFSSolver
from solvers.distance_vision_genetic_solver_with_fallback import DistanceVisionGeneticSolverWithFallback
from solvers.hamilton_solver import HamiltonSolver
from solvers.random_solver import RandomSolver


class GameProvider:

    def __init__(self):
        self.random_solver = RandomSolver()
        self.survival_random_solver = solvers.survival_random_solver.SurvivalRandomSolver()
        self.dfs_solver = DFSSolver()
        self.hamilton_solver = HamiltonSolver()
        self.reward_based_dnn_solver = solvers.reward_based_dnn_solver.RewardBasedDnnSolver()
        self.short_vision_genetic = solvers.short_vision_genetic_solver.ShortVisionGeneticSolver()
        self.distance_vision_genetic = ag.DistanceVisionGeneticSolver()
        self.distance_vision_genetic_with_fallback = DistanceVisionGeneticSolverWithFallback()
        self.all_solvers = [
            self.random_solver,
            self.survival_random_solver,
            self.dfs_solver,
            self.reward_based_dnn_solver,
            self.short_vision_genetic,
            self.distance_vision_genetic,
            self.distance_vision_genetic_with_fallback,
            self.hamilton_solver,
        ]

    def get_n_best(self, games: List[Game], n: int):
        result = sorted(games, key=lambda x: len(x.game_statuses[-1].snake), reverse=True)
        return result[0:n]

    def get_all_game_types(self, n=1, board_size=6, snake_size=4):
        return [GameProvider.get_random_game(solver, board_size, snake_size) for i in range(n) for solver in
                self.all_solvers]

    def get_all_game_types_default(self, n=1):
        return [self._get_default_game(solver) for i in range(n) for solver in self.all_solvers]

    def _get_default_game(self, solver):
        game_seed = game.game_seed_creator.create_default_game_seed()
        game_statuses = solver.solve(game_seed)
        return Game(game_statuses)

    def get_default_games(self, solver, number) -> game.game_seed_creator.List[Game]:
        games = []
        for i in range(number):
            games.append(self._get_default_game(solver))
        return games

    @staticmethod
    def get_random_game(solver, board_size, snake_size):
        game_seed = game.game_seed_creator.create_random_game_seed(board_size, snake_size)
        game_statuses = solver.solve(game_seed)
        return Game(game_statuses)

    @utils.timing.timeit
    def get_random_games(self, solver, number_of_games=1, board_size=6, snake_size=5) -> List[Game]:
        seed()
        games = []
        for i in range(number_of_games):
            print("Solving game={}".format(i))
            games.append(self.get_random_game(solver, board_size=board_size, snake_size=snake_size))
        return games
