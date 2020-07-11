import unittest

import solvers.reward_based_dnn_solver


class TestBasicDnnSolver(unittest.TestCase):

    def test_next_dir(self):
        a = solvers.reward_based_dnn_solver.RewardBasedDnnSolver()
        self.assertEqual(1, 1)
