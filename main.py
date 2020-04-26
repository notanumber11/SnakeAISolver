from gui.window import Window
import model.game_provider as game_provider

print("Solving games...")

game_provider = game_provider.GameProvider()
games = game_provider.get_random_games(game_provider.basic_solver, 1, board_size=6, snake_size=2)
#game_provider = game_provider.GameProvider()
#games = game_provider.get_all_game_types()
input("Press Enter to continue...")
print("Creating window...")
window = Window(games)
window.start()