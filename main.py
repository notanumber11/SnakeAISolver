import random

from gui.game_drawer import GameDrawer
from model.game_status import GameStatus

def generate_snake(board_size, snake_size):
    snake = []
    x = random.randint(0, board_size-1)
    y = random.randint(0, board_size-1)
    head = [x, y]
    snake.append(head)
    for i in range(snake_size-1):
        dirs = random.sample(GameStatus.DIRS, len(GameStatus.DIRS))
        for dir in dirs:
            head_new = [head[0] + dir.x, head[1] + dir.y]
            if head_new not in snake and 0 <= head_new[0] < board_size and 0 <= head_new[1] < board_size:
                snake.insert(0, head_new)
                head = head_new
                break
    if len(snake) < snake_size:
        snake = generate_snake(board_size, snake_size)
    return snake

size = 6
snake_start = generate_snake(size, 5)
# snake_start = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
game = GameStatus(size, snake_start)

print("Starting game...")
gameDrawer = GameDrawer(game)

gameDrawer.start()

