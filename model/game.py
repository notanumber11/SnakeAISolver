from model.point import Point


class Game:
    UP = Point(0, -1)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)
    RIGHT = Point(1, 0)

    def __init__(self, size: int, snake: list, apple: list):
        """
        :param size: Number of cells
        :param snake: List of positions of the snake [[x0, y0], [x1, y1], ... , [x2, y2]]
        :param apple: Position of the apple [x, y]
        """
        self.snake = Point.ints_to_points(snake)
        self.apple = Point(apple[0], apple[1])
        self.size = size
        self.head = snake[0]
        if self.size < 3 or not self.is_valid_snake() or not self.is_inside_board(self.apple):
            raise ValueError("The size is less than 3 or the snake or the apple contain invalid values... ")

    def is_valid_snake(self):
        for pos in self.snake:
            if not self.is_inside_board(pos):
                return False
        return len(self.snake) == len(set(self.snake))

    def is_inside_board(self, pos):
        # Out of boundaries
        if pos.x > self.size - 1 or pos.x < 0:
            return False
        if pos.y > self.size - 1 or pos.y < 0:
            return False
        return True

    def can_move_to_pos(self, pos: Point):
        if not self.is_inside_board(pos):
            return False

        head = self.snake[0]
        dir = Point(pos.x - head.x, pos.y - head.y)
        if not Game.is_valid_dir(dir):
            return False

        # Check that we are not hitting ourselves
        # If we are not taking the apple we can move to last position of the snake
        snake_len = len(self.snake) if pos == self.apple else len(self.snake) - 1
        for point in range(snake_len):
            if pos == point:
                return False
        return True

    def can_move_to_dir(self, dir: Point):
        pos = Point(self.snake[0].x + dir.x, self.snake[0].y + dir.y)
        return self.can_move_to_pos(pos)

    def move(self, d: Point):
        if not Game.is_valid_dir(d):
            raise ValueError("Invalid direction: " + str(d))
        head = Point(self.snake[0].x + d.x, self.snake[0].y + d.y)
        self.snake.insert(0, head)
        if head != self.apple:
            self.snake.pop()
        self.head = self.snake[0]

    @staticmethod
    def is_valid_dir(d: Point):
        return d == Game.UP or d == Game.DOWN or d == Game.RIGHT or d == Game.LEFT
