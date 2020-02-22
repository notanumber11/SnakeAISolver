import tkinter as tk

import gui.canvas as gui
from solvers.basic_solver import BasicSolver
from solvers.dfs_solver import DFSSolver
from solvers.hamilton_solver import HamiltonSolver


class GameDrawer:
    def __init__(self, game):
        self.game = game
        self.tile_size = 40
        self.offset = 100
        self.root = tk.Tk()
        width = self.game.size * self.tile_size + 3 * self.offset
        self.canvas = tk.Canvas(self.root, height=width, width=width, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # basic_solver = BasicSolver()
        # self.games = basic_solver.solve(game)
        # dfs_solver = DFSSolver()
        # self.games = dfs_solver.solve(game)
        hamilton_solver = HamiltonSolver()
        self.games = hamilton_solver.solve(game)


    def start(self):
        self.root.after(500, self.render)
        self.root.mainloop()

    def render(self):
        if not self.games:
            print("Game finished...")
            # self.root.destroy()
            return
        self.game = self.games.pop(0)
        self.draw(self.game)
        self.root.after(500//self.game.size, self.render)

    def draw(self, game):
        self.canvas.delete("all")
        gui.create_grid(self.canvas, self.game.size, self.tile_size, self.offset, self.offset)
        # Draw snake
        for el in game.snake:
            id = gui.draw_rectangle(self.canvas, self.tile_size, el.x, el.y, self.offset, self.offset, "yellow")
        gui.draw_rectangle(self.canvas, self.tile_size, game.snake[0].x, game.snake[0].y, self.offset, self.offset, "red")

        # Draw apple
        gui.draw_rectangle(self.canvas, self.tile_size, game.apple.x, game.apple.y, self.offset, self.offset, "green")
            # canvas.tag_bind(id, "<Button-1>", lambda u: rectangle_clicked())
        pass
