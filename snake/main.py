from gui.window import Window
import game.game_provider as game_provider

print("Solving games...")

game_provider = game_provider.GameProvider()
games = game_provider.get_all_game_types()
input("Press Enter to continue...")
print("Creating window...")
window = Window(games)
window.start()
