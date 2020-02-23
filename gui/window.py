import tkinter as tk


class Window:

    def __init__(self):
        # Configurable parameters
        self.game_size = 6
        self.tile_size = 40
        self.games_per_row = 3
        self.number_of_games = self.get_number_games()
        # Derived parameters
        self.offsetX = 3 * self.tile_size
        self.offsetY = 3 * self.tile_size
        self.single_snake_size = self.game_size * self.tile_size + 2 * self.offsetX
        self.width = self.single_snake_size * self.games_per_row
        self.canvas = tk.Canvas(self.root, height=600, width=self.width, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def start(self):

    def get_number_games(self):
        return 2
