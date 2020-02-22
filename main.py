from gui.game_drawer import GameDrawer
from model.game import Game

size = 5
snake_start = [[2, 2], [3, 2], [4, 2]]
snake_left = [[1, 2], [2, 2], [3, 2]]
snake_down = [[1, 3], [1, 2], [2, 2]]
apple = [4, 4]
game = Game(size, snake_start, apple)

gameDrawer = GameDrawer(game)

gameDrawer.start()