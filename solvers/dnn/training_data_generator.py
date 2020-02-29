import csv
import math

import constants
from model.game_seed_creator import create_game_seed
from model.game_status import GameStatus
import model.game_provider as game_provider
from solvers.dfs_solver import DFSSolver
from solvers.random_solver import RandomSolver
import pandas as pd


def is_apple_closer(current: GameStatus, next: GameStatus):
    dist_curr = math.sqrt((current.head.x - current.apple.x) ** 2 + (current.head.y - current.apple.y) ** 2)
    dist_next = math.sqrt((next.head.x - next.apple.x) ** 2 + (next.head.y - next.apple.y) ** 2)
    return dist_next < dist_curr


def normalize(val, min, max):
    return (val - min) / (max - min)


def generate_training_data(grid_size, samples=100):
    input_features = ["up", "down", "left", "right", "up available", "down available", "left available", "right available",
                "angle to apple", "reward"]
    output = {
        "eating": 0.7,
        "closer": 0.1,
        "further": -0.2,
        "die": -1.0
    }
    solver = RandomSolver()
    # solver = DFSSolver()
    training_data = []
    while len(training_data) < samples:
        game = game_provider.get_random_game(solver, grid_size)
        for i in range(1, len(game.game_statuses)):
            # Computations based on current game
            current = game.game_statuses[i - 1]
            if not current.is_valid_game():
                continue
            up_available = 1.0 if current.can_move_to_dir(GameStatus.UP) else 0.0
            down_available = 1.0 if current.can_move_to_dir(GameStatus.DOWN) else 0.0
            left_available = 1.0 if current.can_move_to_dir(GameStatus.LEFT) else 0.0
            right_available = 1.0 if current.can_move_to_dir(GameStatus.RIGHT) else 0.0
            angle = normalize(current.get_angle(current.apple, current.head), 0, 6.28)

            # Computations based on next game
            next_ = game.game_statuses[i]
            dir_ = next_.prev_dir
            up = 1.0 if dir_ == GameStatus.UP else 0.0
            down = 1.0 if dir_ == GameStatus.DOWN else 0.0
            left = 1.0 if dir_ == GameStatus.LEFT else 0.0
            right = 1.0 if dir_ == GameStatus.RIGHT else 0.0
            reward = None
            if not next_.is_valid_game():
                reward = output["die"]
            elif next_.head == current.apple:
                reward = output["eating"]
            else:
                if is_apple_closer(current, next_):
                    reward = output["closer"]
                elif not is_apple_closer(current, next_):
                    reward = output["further"]
            if not reward:
                raise ValueError("Result calculated incorrectly...")
            row = [up, down, left, right, up_available, down_available, left_available, right_available, angle, reward]
            if reward == output["die"]:
                for i in range(4):
                    if row[i] == 1 and row[i+4] == 1:
                        next_.is_valid_game()
                        raise ValueError("Something wrong with game state")
            training_data.append(
                row)

    print("Training data has been generated...")
    create_csv(input_features, training_data)
    # To test we try with each dir for the neural net and take the one with better output


def create_csv(labels, data):
    with open(constants.DATA_DIR + "training_data.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(data)
    pass


generate_training_data(6, 1000)
