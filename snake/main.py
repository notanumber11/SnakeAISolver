import os
import sys

from gui.gui_starter import game, show_games
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
        from game.game_provider import GameProvider

        game_provider = GameProvider()
        finished = False
        snake_size = 2
        board_size = 18
        i = 0
        solver = game_provider.distance_vision_genetic_with_fallback
        while not finished:
            i += 1
            game = game_provider.get_random_game(solver, board_size, snake_size)
            result = len(game.game_statuses[-1].snake)
            print("Try number={} with result={}".format(i, result))
            if result == board_size * board_size:
                finished = True
        show_games([game])
