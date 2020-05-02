import unittest

from model.point import Point


class TestPoint(unittest.TestCase):

    def test_equals(self):
        point1 = Point(5, 7)
        point2 = Point(5, 7)
        point3 = Point(6, 7)
        point4 = Point(5, 8)
        point5 = None
        self.assertTrue(point1 == point2)
        self.assertEqual(point1, point2)
        self.assertFalse(point1 == point3)
        self.assertNotEqual(point1, point3)
        self.assertNotEqual(point1, point4)
        self.assertNotEqual(point1, point5)

    def test_values(self):
        point = Point(1, 2)
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 2)
        self.assertNotEqual(point.x, 99)
        self.assertNotEqual(point.y, 99)

    def test_hash(self):
        p1 = Point(1, 1)
        p2 = Point(1, 1)
        p3 = Point(1, 2)
        s1 = {p1, p2}
        self.assertEqual(1, len(s1))
        s2 = {p1, p2, p3}
        self.assertEqual(2, len(s2))
