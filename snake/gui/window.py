import math
import tkinter as tk
from typing import List

from game.game import Game
from gui.game_drawer import GameDrawer


class Window:

    def __init__(self, games: List[Game]):
        # Configurable parameters
        self.games = games
        self.game_size = games[0].game_statuses[0].size
        self.tile_size = 40
        self.cols = 3
        self.rows = math.ceil(len(self.games) / self.cols)
        # Derived parameters
        self.cols = min(self.cols, len(games))
        self.grid_size = self.game_size * self.tile_size
        self.offset_x = self.tile_size // 2
        self.offset_y = self.offset_x
        self.game_width = self.grid_size + 2 * self.offset_x
        self.game_height = self.grid_size + 2 * self.offset_y
        self.width = self.game_width * self.cols
        self.height = self.game_height * self.rows
        # Tk configuration
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.game_drawer = GameDrawer()
        self.movements_per_second = 16
        self.should_close_automatically = 0
        self.root.bind('<Return>', self.mark_as_ready)
        self.is_ready = False
        self.first = True

    def mark_as_ready(self, event):
        self.is_ready = True

    def draw(self, games: List[Game]):
        if self.is_ready or self.first:
            self.canvas.delete("all")
            self.first = False
            for pos, game in enumerate(games):
                i = pos // self.cols
                j = pos % self.cols
                offset_x = self.offset_x + j * self.game_width
                offset_y = self.offset_y + i * self.game_width
                self.game_drawer.draw(self.canvas, game, self.game_size, self.tile_size,
                                      offset_x, offset_y)

        self.root.after(int(1000 / self.movements_per_second), lambda: self.draw(self.games))
        if all(game.is_finished() for game in games) and self.should_close_automatically != 0:
            self.root.after(self.should_close_automatically, lambda: self.end())

    def start(self):
        self.root.after(500, lambda: self.draw(self.games))
        self.root.lift()
        self.root.mainloop()

    def end(self):
        self.root.destroy()
