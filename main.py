from gui.window import Window
from model.game_provider import get_games

print("Creating window...")

games = get_games()
# input("Press Enter to continue...")
window = Window(games)
window.start()
