import os
from typing import List

from game.game_provider import GameProvider
from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated
from utils.snake_logger import get_module_logger

LOGGER = get_module_logger(__name__)


def get_models_from_path(path) -> List[str]:
    if path is None:
        return None
    path = os.path.normpath(path)
    subfolders = [str(f.path) for f in os.scandir(path) if f.is_dir()]
    subfolders = [s.replace("\\", "/") for s in subfolders]
    subfolders.sort(key=lambda folder: float(folder.split("/")[-1].split("_")[0]))
    subfolders = [os.path.normpath(s) for s in subfolders]
    return subfolders


def show_solver(solver, board_size, snake_size, number_of_games=6, number_of_tries=None):
    from gui.window import Window
    LOGGER.info("Showing solver...")
    game_provider = GameProvider()
    if number_of_tries == None:
        number_of_tries = number_of_games
    games = game_provider.get_random_games(solver, number_of_tries, board_size, snake_size)
    games = game_provider.get_n_best(games, number_of_games)
    for game in games:
        ga = ModelGeneticEvaluated([game], None)
        print("The snake reached: {}".format(len(game.game_statuses[-1].snake)) + "with fitness: {} ".format(
            ga.fitness()))
    LOGGER.info("Creating window...")
    input("Press Enter to continue...")
    window = Window(games)
    window.should_close_automatically = 3000
    window.start()


def game():
    from gui.window import Window
    from game.game_provider import GameProvider
    LOGGER.info("Solving games...")
    game_provider = GameProvider()
    games = game_provider.get_all_game_types(1, 6, 4)

    input("Press Enter to continue...")
    LOGGER.info("Creating window...")
    window = Window(games)
    window.start()
