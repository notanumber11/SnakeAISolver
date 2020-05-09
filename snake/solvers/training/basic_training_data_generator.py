import math

import solvers.training.training_utils

import game.game_provider
from game.game_status import GameStatus
from solvers.random_solver import RandomSolver

DATA_DIR = "../data/basic_dnn/"
TRAINING_DATA_BASIC_DNN = "training_data_basic_dnn"
LABELS = ["up", "down", "left", "right", "up available", "down available", "left available", "right available",
          "angle to apple", "reward"]


def generate_random_training_data(grid_size, snake_size, samples: int = 100):
    """
    Generate number of samples for training with the following format:
    ["up", "down", "left", "right", "up available", "down available", "left available", "right available", "angle to apple", "reward"]
    Example value when decision is go up and we eat the apple in that movement:
    [1, 0, 0, 0, 1, 1, 0, 0, 0.8, 0.7]
    """
    solver = RandomSolver()
    # game_provider = GameProvider()
    training_data = []
    enough_samples = False
    game_provider = game.game_provider.GameProvider()
    while not enough_samples:
        _game = game_provider.get_random_game(solver, grid_size, snake_size)
        for i in range(1, len(_game.game_statuses)):
            # Computations based on current game status
            current = _game.game_statuses[i - 1]
            if not current.is_valid_game():
                continue
            up_available = 1.0 if current.can_move_to_dir(GameStatus.UP) else 0.0
            down_available = 1.0 if current.can_move_to_dir(GameStatus.DOWN) else 0.0
            left_available = 1.0 if current.can_move_to_dir(GameStatus.LEFT) else 0.0
            right_available = 1.0 if current.can_move_to_dir(GameStatus.RIGHT) else 0.0
            angle = solvers.training.training_utils.normalize_rad_angle(current.get_angle(current.apple, current.head))

            # Computations based on decision taken
            next_ = _game.game_statuses[i]
            dir_ = next_.prev_dir
            up = 1.0 if dir_ == GameStatus.UP else 0.0
            down = 1.0 if dir_ == GameStatus.DOWN else 0.0
            left = 1.0 if dir_ == GameStatus.LEFT else 0.0
            right = 1.0 if dir_ == GameStatus.RIGHT else 0.0
            reward = get_reward(current, next_)
            row = [up, down, left, right, up_available, down_available, left_available, right_available, angle, reward]
            training_data.append(row)
            if len(training_data) >= samples:
                enough_samples = True
                break

    print("Training data has been generated...")
    solvers.training.training_utils.create_csv(LABELS, training_data,
                                               TRAINING_DATA_BASIC_DNN)


def get_input_from_game_status(game_status: GameStatus):
    """
    The goal of this method is to create 4 inputs (one per direction) with the following data
    ["up", "down", "left", "right", "up available", "down available", "left available", "right available", "angle to apple"]
    Example for up:
    [1, 0, 0, 0, 1, 1, 0, 0, 0.8]
    """
    angle = solvers.training.training_utils.normalize_rad_angle(
        game_status.get_angle(game_status.apple, game_status.head))
    angle = round(angle, 2)
    available = [1 if game_status.can_move_to_dir(d) else 0 for d in GameStatus.DIRS]
    inputs = []
    for i in range(len(GameStatus.DIRS)):
        _input = [0] * 4
        _input[i] = 1
        inputs.append(_input)
    for _input in inputs:
        _input += available
        _input.append(angle)
    return inputs


def _is_apple_closer(current: GameStatus, next: GameStatus):
    dist_curr = math.sqrt((current.head.x - current.apple.x) ** 2 + (current.head.y - current.apple.y) ** 2)
    dist_next = math.sqrt((next.head.x - next.apple.x) ** 2 + (next.head.y - next.apple.y) ** 2)
    return dist_next < dist_curr


def get_reward(current, next_):
    reward_values = {
        "eating": 0.7,
        "closer": 0.1,
        "further": -0.2,
        "die": -1.0
    }
    reward = None
    if not next_.is_valid_game():
        reward = reward_values["die"]
    elif next_.head == current.apple:
        reward = reward_values["eating"]
    else:
        if _is_apple_closer(current, next_):
            reward = reward_values["closer"]
        elif not _is_apple_closer(current, next_):
            reward = reward_values["further"]
    if not reward:
        raise ValueError("Reward calculated incorrectly...")
    return reward
