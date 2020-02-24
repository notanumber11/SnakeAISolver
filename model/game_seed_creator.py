import random
from typing import List

from model.game_status import GameStatus


def create_default_game_seed() -> GameStatus:
    snake = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
    apple = [0, 0]
    board_size = 6
    game_seed = GameStatus(board_size, snake, apple)
    return game_seed


def create_game_seed(board_size: int, snake_size: int) -> GameStatus:
    snake = _generate_snake(board_size, snake_size)
    game_seed = GameStatus(board_size, snake)
    return game_seed


def _generate_snake(board_size: int, snake_size: int) -> List[List[int]]:
    snake = []
    x = random.randint(0, board_size - 1)
    y = random.randint(0, board_size - 1)
    head = [x, y]
    snake.append(head)
    for i in range(snake_size - 1):
        dirs = random.sample(GameStatus.DIRS, len(GameStatus.DIRS))
        for dir_ in dirs:
            head_new = [head[0] + dir_.x, head[1] + dir_.y]
            if head_new not in snake and 0 <= head_new[0] < board_size and 0 <= head_new[1] < board_size:
                snake.insert(0, head_new)
                head = head_new
                break
    if len(snake) < snake_size:
        snake = _generate_snake(board_size, snake_size)
    return snake
