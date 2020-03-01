import math
import time

from model.game import Game
from model.game_seed_creator import *
from solvers.basic_dnn.basic_dnn_solver import BasicDnnSolver
from solvers.dfs_solver import DFSSolver
from solvers.hamilton_solver import HamiltonSolver
from solvers.random_solver import RandomSolver

dfs_solver = DFSSolver()
hamilton_solver = HamiltonSolver()
random_solver = RandomSolver()
basic_dnn_solver = BasicDnnSolver()

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_games():
    result = get_random_games(basic_dnn_solver, 10)
    return result

def get_n_best(games: List[Game], n: int):
    result = sorted(games, key=lambda x: len(x.game_statuses[-1].snake), reverse=True)
    return result[0:n]

def get_all_game_types():
    return [get_default_game(random_solver), get_default_game(dfs_solver), get_default_game(hamilton_solver)]


def get_default_game(solver):
    game_seed = create_default_game_seed()
    game_statuses = solver.solve(game_seed)
    return Game(game_statuses)


def get_default_games(solver, number) -> List[Game]:
    games = []
    for i in range(number):
        games.append(get_default_game(solver))
    return games


def get_random_game(solver, board_size, snake_size=None):
    if not snake_size:
        snake_size = random.randint(2, board_size)
    game_seed = create_game_seed(board_size, snake_size)
    game_statuses = solver.solve(game_seed)
    return Game(game_statuses)


@timeit
def get_random_games(solver, number) -> List[Game]:
    games = []
    for i in range(number):
        games.append(get_random_game(solver, 6, 5))
    return games
