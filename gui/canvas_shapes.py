def create_grid(canvas, size, tile_size, offset_x, offset_y):
    width = tile_size * size
    height = tile_size * size
    canvas.delete('grid_line')  # Will only remove the grid_line
    # Creates all vertical lines at intervals of tile_size
    for i in range(0, width + 1, tile_size):
        x0 = offset_x + i
        y0 = offset_y
        x1 = x0
        y1 = offset_y + height
        canvas.create_line([(x0, y0), (x1, y1)])
    # Creates all horizontal lines at intervals of tile_size
    for i in range(0, height + 1, tile_size):
        x0 = offset_x
        y0 = offset_y + i
        x1 = offset_x + width
        y1 = y0
        canvas.create_line([(x0, y0), (x1, y1)])


def draw_rectangle(canvas, size, i, j, offset_x, offset_y, color):
    x = offset_x + i * size
    y = offset_y + j * size
    x1 = x + size
    y1 = y + size
    id = canvas.create_rectangle(x, y, x1, y1, fill=color)
    return id
