import os
import sys

print("*******************************************")
print("Starting to run main.py")
print(sys.version)
print("The current directory is: {}".format(os.getcwd()))
from utils import aws_snake_utils
print("The running environment is: {}".format(aws_snake_utils.get_running_environment()))
print("*******************************************")

import argparse


def game():
    from gui.window import Window
    from game.game_provider import GameProvider
    print("Solving games...")
    game_provider = GameProvider()
    games = game_provider.get_all_game_types()
    input("Press Enter to continue...")
    print("Creating window...")
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
        print("Running train_basic_dnn ...")
        from train_basic_dnn import train_basic_dnn
        train_basic_dnn()
    if args.type == "train_basic_genetic":
        print("Running train_basic_genetic ...")
        from train_genetic_algorithm import train_basic_genetic
        train_basic_genetic()
    if args.type == "train_advanced_genetic":
        print("Running train_advance_genetic ...")
        from train_genetic_algorithm import train_advance_genetic
        train_advance_genetic()
