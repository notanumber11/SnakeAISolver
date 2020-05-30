import unittest
import matplotlib.pyplot as plt
import pandas as pd

from game.game import Game
from game.game_status import GameStatus
from solvers.genetic.advance_genetic_model_evaluated import AdvanceModelGeneticEvaluated


class TestAdvanceGeneticModelEvaluated(unittest.TestCase):

    def test_me_parameters(self):
        snake = [[1, 0], [2, 0]]
        apple = [0, 0]
        game_status = GameStatus(6, snake, apple)
        new_game_status = game_status.move(GameStatus.LEFT)
        game = Game([game_status, new_game_status])
        me = AdvanceModelGeneticEvaluated([game], None)
        self.assertEqual(1, me.movements)
        self.assertEqual(3, me.snake_length)
        self.assertEqual(1, me.apples)

    def test_me_parameters_when_loop(self):
        snake = [[1, 0], [2, 0]]
        apple = [0, 0]
        game_status = GameStatus(6, snake, apple)
        new_game_status = game_status.move(GameStatus.LEFT)
        game = Game([game_status, new_game_status])
        game.was_stack_in_loop = True
        me = AdvanceModelGeneticEvaluated([game], None)
        self.assertEqual(1, me.movements)
        self.assertEqual(0, me.snake_length)
        self.assertEqual(0, me.apples)
        self.assertEqual(6, me.size)
        self.assertLessEqual(me.fitness(), 10)

    def test_research_fitness_function(self):
        snake = [[1, 0], [2, 0]]
        apple = [0, 0]
        game_status = GameStatus(6, snake, apple)
        new_game_status = game_status.move(GameStatus.LEFT)
        game = Game([game_status, new_game_status])
        me = AdvanceModelGeneticEvaluated([game], None)
        # At the beginning is more important to move a bit without die than to take apples
        one_apple_one_movement = me.fitness()
        me.movements = 5
        one_apple_five_movements = me.fitness()
        self.assertGreater(one_apple_five_movements, one_apple_one_movement)
        # Taking the same number of apples with lower movements is better when you already have some apples
        me.movements = 10
        me.apples = 10
        ten_apples_ten_movements = me.fitness()
        me.movements = 20
        ten_apples_twenty_movements = me.fitness()
        self.assertGreater(ten_apples_ten_movements, ten_apples_twenty_movements)

    def test_fitness_value(self):
        snake = [[1, 0], [2, 0]]
        apple = [0, 0]
        game_status = GameStatus(11, snake, apple)
        new_game_status = game_status.move(GameStatus.LEFT)
        game = Game([game_status, new_game_status])
        me = AdvanceModelGeneticEvaluated([game], None)
        me.movements = 2000
        me.apples = 19
        print(me.fitness())

