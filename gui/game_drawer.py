import tkinter as tk

import gui.canvas as gui

class GameDrawer:
    def __init__(self, game):
        self.game = game
        self.tile_size = 40
        self.offset = 100
        self.root = tk.Tk()
        width = self.game.size * self.tile_size + 3 * self.offset
        self.canvas = tk.Canvas(self.root, height=width, width=width, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def start(self):
        self.draw(self.game)
        # root.after(2000, task)
        self.root.mainloop()

    def draw(self, game):
        self.canvas.delete("all")
        gui.create_grid(self.canvas, self.game.size, self.tile_size, self.offset, self.offset)
        for el in game.snake:
            id = gui.draw_rectangle(self.canvas, self.tile_size, el.x, el.y, self.offset, self.offset)
            # canvas.tag_bind(id, "<Button-1>", lambda u: rectangle_clicked())
        pass


