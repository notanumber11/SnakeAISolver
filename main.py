from gui.game_drawer import GameDrawer
from model.game import Game

size = 6
snake_start = [[2, 2], [3, 2]]
apple = [1, 2]
game = Game(size, snake_start, apple)

gameDrawer = GameDrawer(game)

gameDrawer.start()
