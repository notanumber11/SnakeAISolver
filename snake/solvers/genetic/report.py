import csv
import os
import shutil
from typing import List

from solvers.genetic.hyperparameters import HyperParameters
from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated
from utils.snake_logger import get_module_logger

LOGGER = get_module_logger(__name__)


class Report:
    REPORT_NAME = "report.csv"
    HYPERPARAMETERS_NAME = "hyperparameters.json"
    LABELS = ["Iteration", "BestCompletion", "AvgCompletion", "BestApples", "AvgApples", "BestMovs", "AvgMovs", "BestFitness", "AvgFitness", "Time"]

    def __init__(self, path: str, h: HyperParameters):
        self.create_folder(path)
        self.save_hyperparameters(h, path)
        self.report_path = self.start_report(path)
        self.h = h
        self.results = []

    def start_report(self, path):
        report_path = os.path.normpath(path + Report.REPORT_NAME)
        LOGGER.info("Saving report to: {}".format(report_path))
        with open(report_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows([Report.LABELS])
        return report_path

    def save_hyperparameters(self, h, path):
        hyperparameters_path = os.path.normpath(path + Report.HYPERPARAMETERS_NAME)
        LOGGER.info("Saving hyperparameters to: {}".format(hyperparameters_path))
        with open(hyperparameters_path, "w") as file:
            file.write(h.__str__())

    def create_folder(self, path):
        folder_path = os.path.normpath(path)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)

    def generate_report(self, iteration, previous_top_population: List[ModelGeneticEvaluated], time):
        # Best measures
        best = previous_top_population[0]
        best_completion = (best.apples + self.h.snake_size) / self.h.game_size ** 2
        best_apples = best.apples
        best_movs = best.movements
        best_fitness = best.fitness()

        # Population average measures
        avg_apples = 0
        avg_movs = 0
        avg_fitness = 0
        len_ = len(previous_top_population)
        for el in previous_top_population:
            avg_apples += el.apples
            avg_movs += el.movements
            avg_fitness += el.fitness()
        avg_apples = avg_apples / len_
        avg_movs = avg_movs / len_
        avg_fitness = avg_fitness / len_
        avg_completion = (avg_apples + self.h.snake_size) / self.h.game_size ** 2
        vals = [iteration, best_completion, avg_completion, best_apples, avg_apples, best_movs, avg_movs, best_fitness, avg_fitness, time]
        vals = [round(x, 2) for x in vals]
        self.results.append(vals)
        assert(len(vals) == len(Report.LABELS))
        with open(self.report_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(vals)
