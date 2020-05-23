import os

from game import game_provider
from game.game_provider import GameProvider
from solvers.genetic.advance_genetic_solver import AdvanceGeneticSolver
from utils.snake_logger import get_module_logger

LOGGER = get_module_logger(__name__)


def get_last_model_from_path(path):
    subfolders = [str(f.path) for f in os.scandir(path) if f.is_dir()]
    subfolders.sort(key=lambda folder: float(folder.split("\\")[-1].split("_")[0]))
    return subfolders[-1]


def show_solver(solver, board_size, snake_size, number_of_games=6):
    from gui.window import Window
    LOGGER.info("Showing solver...")
    game_provider = GameProvider()
    games = game_provider.get_random_games(solver, number_of_games,board_size,snake_size)
    LOGGER.info("Creating window...")
    window = Window(games)
    window.should_close_automatically = 3000
    window.start()

def game():
    from gui.window import Window
    from game.game_provider import GameProvider
    LOGGER.info("Solving games...")
    game_provider = GameProvider()
    games = game_provider.get_all_game_types()

    input("Press Enter to continue...")
    LOGGER.info("Creating window...")
    window = Window(games)
    window.start()