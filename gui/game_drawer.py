import time
import timeit
import tkinter as tk

import gui.canvas as gui
from solvers.basic_solver import BasicSolver
from solvers.dfs_solver import DFSSolver
from solvers.hamilton_solver import HamiltonSolver


class GameDrawer:
    def __init__(self, game):
        # Size parameters
        self.tile_size = 40
        self.offset = 100
        self.first_render = True
        self.game = game
        self.games = []
        self.root = tk.Tk()
        width = self.game.size * self.tile_size + 3 * self.offset
        self.canvas = tk.Canvas(self.root, height=width, width=width, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        basic_solver = BasicSolver()
        dfs_solver = DFSSolver()
        hamilton_solver = HamiltonSolver()
        all_solutions = []
        self.solve_with_timer(dfs_solver, all_solutions)
        self.games = all_solutions[0]

    def solve_with_timer(self, solver, solutions, executions=5):
        time = timeit.timeit(lambda: self.solve(solver, solutions), number=executions)
        print("{}: Total time (s): {:.2f} Number of executions: {} Avg: {:.2f}".format(solver, time, executions, time / executions))

    def solve(self, solver, solutions):
        print(len(solutions))
        solution = solver.solve(self.game)
        solutions.append(solution)
        return solution

    def start(self):
        self.root.after(50, self.render)
        self.root.mainloop()

    def render(self):
        if not self.games:
            print("Game finished...")
            time.sleep(1)
            self.root.destroy()
            return
        self.game = self.games.pop(0)
        self.draw(self.game)
        speed = 500 // self.game.size
        if self.first_render:
            speed = 500
            self.first_render = False
        self.root.after(speed, self.render)

    def draw(self, game):
        self.canvas.delete("all")
        gui.create_grid(self.canvas, self.game.size, self.tile_size, self.offset, self.offset)
        # Draw apple
        if game.apple:
            gui.draw_rectangle(self.canvas, self.tile_size, game.apple.x, game.apple.y, self.offset, self.offset,
                               "green")
        # Draw snake
        for el in game.snake:
            id = gui.draw_rectangle(self.canvas, self.tile_size, el.x, el.y, self.offset, self.offset, "yellow")
        gui.draw_rectangle(self.canvas, self.tile_size, game.snake[0].x, game.snake[0].y, self.offset, self.offset,
                           "red")
        # canvas.tag_bind(id, "<Button-1>", lambda u: rectangle_clicked())

    def end(self):
        self.root.destroy()
