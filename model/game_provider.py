import time

from model.game import Game
from model.game_seed_creator import *
from solvers.dfs_solver import DFSSolver
from solvers.hamilton_solver import HamiltonSolver


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
    return get_all_game_types()


def get_default_game_hamilton():
    game_seed = create_default_game_seed()
    hamilton_solver = HamiltonSolver()
    game_statuses = hamilton_solver.solve(game_seed)
    game = Game(game_statuses)
    return game


def get_all_game_types():
    return [get_default_game_dfs(), get_default_game_hamilton()]


def get_default_game_dfs() -> Game:
    game_seed = create_default_game_seed()
    dfs_solver = DFSSolver()
    game_statuses = dfs_solver.solve(game_seed)
    game = Game(game_statuses)
    return game


def get_random_game_dfs() -> Game:
    game_seed = create_game_seed(6, 5)
    dfs_solver = DFSSolver()
    game_statuses = dfs_solver.solve(game_seed)
    game = Game(game_statuses)
    return game


def get_default_games_dfs() -> List[Game]:
    games = []
    for i in range(9):
        games.append(get_default_game_dfs())
    return games


@timeit
def get_random_games_dfs() -> List[Game]:
    games = []
    for i in range(4):
        games.append(get_random_game_dfs())
    return games
