from game.game_seed_creator import create_default_game_seed
from solvers.advance_genetic.advance_genetic_algorithm import AdvanceGeneticAlgorithm
from tests.solvers.basic_genetic.test_genetic_algorithm import GeneticAlgorithmTest
import numpy as np


class TestAdvanceGeneticAlgorithm(GeneticAlgorithmTest):
    def setUp(self):
        self.layers_size = [28, 20, 12, 4]
        self.ga = AdvanceGeneticAlgorithm(self.layers_size)

    def test_max_number_of_movements(self):
        self.assertTrue(False)

    def test_parallel_processing(self):
        self.assertTrue(False)
