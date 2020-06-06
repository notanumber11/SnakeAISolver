from typing import List

import game.game_seed_creator
import solvers.basic_dnn_solver
import solvers.basic_solver
import solvers.advance_genetic_solver as ag
import solvers.basic_genetic_solver
import utils.timing
from game.game import Game
from solvers.dfs_solver import DFSSolver
from solvers.hamilton_solver import HamiltonSolver
from solvers.random_solver import RandomSolver


class GameProvider:

    def __init__(self):
        self.dfs_solver = DFSSolver()
        self.hamilton_solver = HamiltonSolver()
        self.random_solver = RandomSolver()
        self.basic_solver = solvers.basic_solver.BasicSolver()
        self.basic_dnn = solvers.basic_dnn_solver.BasicDnnSolver()
        self.basic_genetic = solvers.basic_genetic_solver.BasicGeneticSolver()
        self.advance_genetic = ag.AdvanceGeneticSolver()
        self.all_solvers = [
                            self.random_solver,
                            self.basic_solver,
                            self.dfs_solver,
                            self.basic_dnn,
                            self.basic_genetic,
                            self.advance_genetic,
                            # self.hamilton_solver,
        ]

    def get_n_best(self, games: List[Game], n: int):
        result = sorted(games, key=lambda x: len(x.game_statuses[-1].snake), reverse=True)
        return result[0:n]

    def get_all_game_types(self, n=1, board_size=6, snake_size=4):
        return [GameProvider.get_random_game(solver, board_size, snake_size) for i in range(n) for solver in self.all_solvers]

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
    def get_random_game(solver, board_size, snake_size=None):
        if not snake_size:
            snake_size = game.game_seed_creator.random.randint(2, snake_size)
        game_seed = game.game_seed_creator.create_random_game_seed(board_size, snake_size)
        game_statuses = solver.solve(game_seed)
        return Game(game_statuses)

    @utils.timing.timeit
    def get_random_games(self, solver, number_of_games=1, board_size=6, snake_size=5) -> List[Game]:
        games = []
        for i in range(number_of_games):
            games.append(self.get_random_game(solver, board_size, snake_size))
        return games
