from gui.window import Window
from model.game_provider import get_games

print("Solving games...")

games = get_games()
#input("Press Enter to continue...")
print("Creating window...")
window = Window(games)
window.start()