import os
import sys

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['game', 'train_basic_dnn', 'train_basic_genetic', 'train_advanced_genetic'],
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
