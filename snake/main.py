from gui.window import Window
import model.game_provider as game_provider

print("Solving games...")

game_provider = game_provider.GameProvider()
games = game_provider.get_random_games(game_provider.basic_genetic, 25, board_size=10, snake_size=4)
games = game_provider.get_n_best(games, 1)
#games = game_provider.get_all_game_types(2)
input("Press Enter to continue...")
print("Creating window...")
window = Window(games)
window.start()