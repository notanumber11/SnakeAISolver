import numpy as np


# Based on: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.7460&rep=rep1&type=pdf
# https://github.com/Chrispresso/SnakeAI/blob/f1c6659a216bfc623c99a8cf225f8ae114893d87/genetic_algorithm/crossover.py
def simulated_binary_crossover(model_genetic_father, model_genetic_mother, eta=100):
    child_a, child_b = [], []
    for i in range(len(model_genetic_father)):
        father_chromosome, mother_chromosome = model_genetic_father[i], model_genetic_mother[i]
        u = np.random.random(father_chromosome.shape)
        beta = np.empty(father_chromosome.shape)
        # Eq 4
        beta[u <= 0.5] = (2.0 * u[u <= 0.5]) ** (1.0 / (eta + 1.0))
        beta[u > 0.5] = (1.0 / (2.0 * (1.0 - u[u > 0.5]))) ** (1.0 / (eta + 1.0))
        # Eq 5
        chromosome_a = 0.5 * ((1.0 + beta) * father_chromosome + (1.0 - beta) * mother_chromosome)
        # Eq 6
        chromosome_b = 0.5 * ((1.0 - beta) * father_chromosome + (1.0 + beta) * mother_chromosome)
        child_a.append(chromosome_a)
        child_b.append(chromosome_b)
    return [child_a, child_b]


def random_crossover(model_genetic_father, model_genetic_mother):
    child_a, child_b = [], []
    for i in range(len(model_genetic_father)):
        layer_father, layer_mother = model_genetic_father[i], model_genetic_mother[i]
        mask_a = np.random.choice([0, 1], size=layer_father.shape)
        mask_b = np.where(mask_a < 1, 1, 0)
        chromosome_a = layer_father * mask_a + layer_mother * mask_b
        chromosome_b = layer_mother * mask_a + layer_father * mask_b
        child_a.append(chromosome_a)
        child_b.append(chromosome_b)
    return [child_a, child_b]


def single_point_binary_crossover(model_genetic_father, model_genetic_mother):
    child_a, child_b = [], []
    for i in range(len(model_genetic_father)):
        layer_father, layer_mother = model_genetic_father[i], model_genetic_mother[i]
        chromosome_a = layer_father.copy()
        chromosome_b = layer_mother.copy()

        rows, cols = layer_mother.shape
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)

        chromosome_a[:row, :] = layer_mother[:row, :]
        chromosome_b[:row, :] = layer_father[:row, :]

        chromosome_a[row, :col] = layer_mother[row, :col]
        chromosome_b[row, :col] = layer_father[row, :col]
        child_a.append(chromosome_a)
        child_b.append(chromosome_b)
    return [child_a, child_b]

