class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if other is None:
            return False
        if self.x != other.x or self.y != other.y:
            return False
        return True

    def __hash__(self):
        return int(str(self.x) + str(self.y))

    def __str__(self):
        return "[{}, {}]".format(self.x, self.y)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def ints_to_points(nums):
        points = []
        for el in nums:
            points.append(Point(el[0], el[1]))
        return points

    @staticmethod
    def points_to_ints(points):
        nums = []
        for p in points:
            nums.append([p.x, p.y])
        return nums
