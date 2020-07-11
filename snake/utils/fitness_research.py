import matplotlib.pyplot as plt
import pandas as pd

from game.game import Game
from game.game_status import GameStatus
from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated


# The goal of this class is to research the best fitness values for the AdvanceModelGeneticEvaluated
class FitnessResearch:

    def __init__(self):
        pass

    def research_fitness_function(self):
        snake = [[1, 0], [2, 0]]
        apple = [0, 0]
        game_status = GameStatus(6, snake, apple)
        game = Game([game_status])
        advance_model_evaluated = ModelGeneticEvaluated([game], None)
        apples = [1, 2, 5, 10, 15, 20]
        movements = [1, 5, 10, 20, 30]
        self.fitness_grow(apples, movements, len(game_status.snake), advance_model_evaluated)

    def fitness_grow(self, apples, movements, initial_size,
                     advance_model: ModelGeneticEvaluated):

        apples = apples
        advance_fitness_dict = {}
        basic_fitness_dict = {}
        other_fitness_dict = {}
        for mov in movements:
            advance_fitness_dict["movs_" + str(mov)] = []
            basic_fitness_dict["movs_" + str(mov)] = []
            other_fitness_dict["movs_" + str(mov)] = []

        for eaten_apples in apples:
            for mov in movements:
                advance_model.apples = eaten_apples
                advance_model.movements = mov

                # Advance fitness
                advance_fitness = advance_model.fitness()
                advance_fitness_dict["movs_" + str(mov)] += [advance_fitness]
                # Basic fitness
                basic_fitness = advance_model.basic_fitness()
                basic_fitness_dict["movs_" + str(mov)] += [basic_fitness]
                # Other fitness
                other_fitness = advance_model.other_fitness()
                other_fitness_dict["movs_" + str(mov)] += [other_fitness]

                # Print all results
                print("Apples={} - Movements={} - \n "
                      "\tAdvance Fitness={:.2f} \n "
                      "\tOther Fitness={:-2f}\n"
                      "\tBasic Fitness={:.2f} \n "
                      .format(eaten_apples, mov, advance_fitness, other_fitness, basic_fitness))
        # Graphs
        self.plot(apples, advance_fitness_dict, "Advance fitness")
        self.plot(apples, other_fitness_dict, "Other fitness")
        self.plot(apples, basic_fitness_dict, "Basic fitness")

    def plot(self, x_axis, fitness_dict, title):
        dataframe = pd.DataFrame(
            fitness_dict,
            index=x_axis)
        ax = dataframe.plot.line(logy=True, title=title)
        ax.set_xticks(x_axis)
        plt.show()


if __name__ == '__main__':
    research_fitness = FitnessResearch()
    research_fitness.research_fitness_function()
