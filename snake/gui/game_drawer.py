import tkinter as tk

import gui.canvas_shapes as canvas_shapes
from game.game import Game


class GameDrawer:
    def __init__(self):
        pass

    def draw(self, canvas: tk.Canvas, game: Game, game_size: int, tile_size: int, offset_x: int, offset_y: int):
        canvas_shapes.create_grid(canvas, game_size, tile_size, offset_x, offset_y)
        game_status = game.next()
        # Draw apple
        if game_status.apple:
            canvas_shapes.draw_rectangle(canvas, tile_size, game_status.apple.x, game_status.apple.y, offset_x,
                                         offset_y,
                                         "red")
        # Draw snake
        for el in game_status.snake:
            id = canvas_shapes.draw_rectangle(canvas, tile_size, el.x, el.y, offset_x, offset_y, "green")
        canvas_shapes.draw_rectangle(canvas, tile_size, game_status.snake[0].x, game_status.snake[0].y, offset_x,
                                     offset_y,
                                     "yellow")
        # canvas.tag_bind(id, "<Button-1>", lambda u: rectangle_clicked())
