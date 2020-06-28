import os
import sys

from gui.gui_starter import get_models_from_path, game
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
    parser.add_argument('type',
                        choices=['game', 'train_basic_dnn', 'train_genetic', 'best'],
                        type=str.lower)
    parser.add_argument('-p', '--checkpoint_path', action='store',
                        help="if path is supplied the model is loaded from there")
    # To not fail with the default argument train provided by AWS
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html
    args, unknown = parser.parse_known_args()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None:
        checkpoint_path = os.path.normpath(checkpoint_path)

    if args.type == "game":
        game()
    if args.type == "train_basic_dnn":
        LOGGER.info("Running train_basic_dnn ...")
        from solvers.basic_dnn.train_basic_dnn import train_basic_dnn
        train_basic_dnn()

    if args.type == "train_genetic":
        LOGGER.info("Running genetic train ...")
        from solvers.genetic.train_genetic_algorithm import train_genetic
        train_genetic(checkpoint_path)

    if args.type == "best":
        from solvers.distance_vision_genetic_solver import DistanceVisionGeneticSolver
        from gui.gui_starter import show_solver

        solver = DistanceVisionGeneticSolver(get_models_from_path(checkpoint_path)[-1])
        show_solver(solver, board_size=10, snake_size=6, number_of_games=1, number_of_tries=5)
