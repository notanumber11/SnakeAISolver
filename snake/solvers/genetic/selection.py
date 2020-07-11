from typing import List

from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated


def elitism_selection(percentage: float, population_evaluated: List[ModelGeneticEvaluated]) -> List[
    ModelGeneticEvaluated]:
    population_evaluated.sort(key=lambda x: x.fitness(), reverse=True)
    number_of_top_performers = int(percentage * len(population_evaluated))
    top_performers = population_evaluated[0:number_of_top_performers]
    return top_performers
