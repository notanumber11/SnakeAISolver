import math
import random

from model.point import Point


class GameStatus:
    UP = Point(0, -1)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)
    RIGHT = Point(1, 0)

    DIRS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, size: int, snake: list, apple: list = None):
        """
        :param size: Number of cells
        :param snake: List of positions of the snake [[x0, y0], [x1, y1], ... , [x2, y2]]
        :param apple: Position of the apple [x, y]
        """
        self.size = size
        self.snake = Point.ints_to_points(snake)
        self.head = self.snake[0]
        self.prev_dir = Point(self.head.x - self.snake[1].x, self.head.y - self.snake[1].y)
        if not self.is_full_finished():
            self.apple = Point(apple[0], apple[1]) if apple is not None else self.generate_new_apple()
            self.angle_to_apple = self.get_angle(self.head, self.apple)
        else:
            self.apple = None
            self.angle_to_apple = 0

    def is_valid_game(self):
        if not self.is_inside_board(self.apple):
            return False
        if self.size < 3:
            return False
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
        if not GameStatus.is_valid_dir(dir):
            return False

        # Check that we are not hitting ourselves
        # If we are not taking the apple we can move to last position of the snake
        snake_len = len(self.snake) if pos == self.apple else len(self.snake) - 1
        for i in range(snake_len):
            if pos == self.snake[i]:
                return False
        return True

    # 0 0 0 0
    # 0 1 2 0
    # 0 0 0 0
    # 0 X 0 0
    #
    # snake = [[2, 1], [1, 1]]
    # head = [2, 1] = Point(2, 1) where head.x = 2 and head.y = 1
    # dir = right = [1, 0] = Point(1, 0) where right.x = 1 and right.y = 0
    # head = [3, 1]
    def can_move_to_dir(self, dir: Point):
        if dir.x + self.prev_dir.x == 0 and dir.y + self.prev_dir.y == 0:
            return False
        head = self.snake[0]
        pos = Point(head.x + dir.x, head.y + dir.y)
        return self.can_move_to_pos(pos)

    def move(self, d: Point):
        if not GameStatus.is_valid_dir(d):
            raise ValueError("Invalid direction: " + str(d))

        # Create a new game status as deep copy
        # The constructor expects as argument lists of integer instead of points
        new_snake = []
        for p in self.snake:
            new_snake.append([p.x, p.y])
        new_apple = [self.apple.x, self.apple.y]
        new_head = [self.snake[0].x + d.x, self.snake[0].y + d.y]
        new_snake.insert(0, new_head)
        # If apple is not eaten remove tail of snake
        if new_head != new_apple:
            new_snake.pop()
        else:
            # If apple is eaten the new GameStatus will be in charge
            # of generating the new apple.
            new_apple = None
        new_game_status = GameStatus(self.size, new_snake, new_apple)
        return new_game_status

    @staticmethod
    def is_valid_dir(d: Point):
        return d == GameStatus.UP or d == GameStatus.DOWN or d == GameStatus.RIGHT or d == GameStatus.LEFT

    def is_full_finished(self):
        grid_size = self.size * self.size
        holes = grid_size - len(self.snake)
        if holes == 0:
            print("Game finished successfully !!!")
            return True
        return False

    def generate_new_apple(self):
        grid_size = self.size * self.size
        hole_pos = random.randint(0, grid_size)
        for i in range(grid_size):
            hole_pos = hole_pos % grid_size
            x = hole_pos % self.size
            y = hole_pos // self.size
            p = Point(x, y)
            if p not in self.snake:
                for s in self.snake:
                    if s.x == p.x and s.y == p.y:
                        raise ValueError("The error is here...")
                return p
            hole_pos += 1
        raise ValueError("Could not find a new position for apple")

    def get_angle(self, head: Point, apple: Point):
        angle = math.atan2(head.y - apple.y, head.x - apple.x)
        return angle


