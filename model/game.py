from random import randint

from model.point import Point


class Game:
    UP = Point(0, -1)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)
    RIGHT = Point(1, 0)

    DIRS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, size: int, snake: list, apple: list):
        """
        :param size: Number of cells
        :param snake: List of positions of the snake [[x0, y0], [x1, y1], ... , [x2, y2]]
        :param apple: Position of the apple [x, y]
        """
        self.snake = Point.ints_to_points(snake)
        self.apple = Point(apple[0], apple[1])
        self.size = size
        self.head = self.snake[0]

    def is_valid_game(self):
        if self.size < 3:
            return False
        if not self.is_inside_board(self.apple):
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
        if not Game.is_valid_dir(dir):
            return False

        # Check that we are not hitting ourselves
        # If we are not taking the apple we can move to last position of the snake
        snake_len = len(self.snake) if pos == self.apple else len(self.snake) - 1
        for i in range(snake_len):
            if pos == self.snake[i]:
                return False
        return True

    def can_move_to_dir(self, dir: Point):
        pos = Point(self.snake[0].x + dir.x, self.snake[0].y + dir.y)
        return self.can_move_to_pos(pos)

    def move(self, d: Point):
        if not Game.is_valid_dir(d):
            raise ValueError("Invalid direction: " + str(d))
        
        new_game = self.clone()
        
        head = Point(new_game.snake[0].x + d.x, new_game.snake[0].y + d.y)
        new_game.snake.insert(0, head)
        # If apple is not eaten remove tail of snake
        if head != new_game.apple:
            new_game.snake.pop()
        else:
            new_game.apple = new_game.generate_new_apple()
        new_game.head = new_game.snake[0]
        return new_game

    @staticmethod
    def is_valid_dir(d: Point):
        return d == Game.UP or d == Game.DOWN or d == Game.RIGHT or d == Game.LEFT

    def generate_new_apple(self):
        holes = self.size * self.size - len(self.snake)
        if holes == 0:
            raise ValueError("There are not places for apple, the game is finished !!!")
        for i in range(self.size * self.size):
            x = i % self.size
            y = i // self.size
            p = Point(x, y)
            if p not in self.snake:
                return p
        raise ValueError("Could not find a new position for apple")

    def clone(self):
        snake = Point.points_to_ints(self.snake)
        apple = [self.apple.x, self.apple.y]
        new_game = Game(self.size, snake, apple)
        return new_game
