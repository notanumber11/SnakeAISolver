from gui.window import Window
import game.game_provider as game_provider

print("Solving games...")

game_provider = game_provider.GameProvider()
games = game_provider.get_random_games(game_provider.basic_genetic, 50, 10, 2)
games = game_provider.get_n_best(games, 6)
input("Press Enter to continue...")
print("Creating window...")
window = Window(games)
window.start()
