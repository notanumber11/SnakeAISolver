import os
import sys

from gui.gui_starter import get_last_model_from_path, game
from utils.snake_logger import get_module_logger

LOGGER = get_module_logger(__name__)

LOGGER.info("*******************************************")
LOGGER.info("Starting to run main.py")
LOGGER.info(sys.version)
LOGGER.info("The current directory is: {}".format(os.getcwd()))
LOGGER.info("The args passed are: {}".format(sys.argv[1:]))
from utils import aws_snake_utils

LOGGER.info("The running environment is: {}".format(aws_snake_utils.get_running_environment()))
LOGGER.info("*******************************************")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['game', 'train_basic_dnn', 'train_basic_genetic', 'train_advanced_genetic', 'best'],
                        type=str.lower)
    # To not fail with the default argument train provided by AWS
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html
    args, unknown = parser.parse_known_args()
    if args.type == "game":
        game()
    if args.type == "train_basic_dnn":
        LOGGER.info("Running train_basic_dnn ...")
        from train_basic_dnn import train_basic_dnn
        train_basic_dnn()

    if args.type == "train_basic_genetic":
        LOGGER.info("Running train_basic_genetic ...")
        from train_genetic_algorithm import train_basic_genetic
        train_basic_genetic()

    if args.type == "train_advanced_genetic":
        LOGGER.info("Running train_advance_genetic ...")
        from train_genetic_algorithm import train_advance_genetic

        train_advance_genetic()

    if args.type == "best":
        from solvers.genetic.advance_genetic_solver import AdvanceGeneticSolver
        from gui.gui_starter import show_solver
        solver = AdvanceGeneticSolver(get_last_model_from_path(r"/home/denis/Escritorio/SnakeIA/snake/models/experiments/pop=1001_sel=0.25_mut_0.05_it_10000_games_1_game_size_8/"))
        show_solver(solver, 8, 2, 6)
