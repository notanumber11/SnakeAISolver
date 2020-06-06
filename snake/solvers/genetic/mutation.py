from typing import List

import numpy as np


def gaussian_mutation(model_genetic: List[np.array], mutation_rate, mu=0, sigma=0.2) -> List[np.array]:
    for i in range(len(model_genetic)):
        chromosome = model_genetic[i]
        mutation_array = np.random.random(chromosome.shape) < mutation_rate
        gaussian_values = np.random.normal(mu, sigma, size=chromosome.shape)
        chromosome[mutation_array] += gaussian_values[mutation_array]
    return model_genetic


def uniform_mutation(model_genetic, mutation_rate, low=-1, high=1):
    for layer_index in range(len(model_genetic)):
        chromosome = model_genetic[layer_index]
        mutation_array = np.random.random(chromosome.shape) < mutation_rate
        mutated_values = np.random.uniform(low, high, size=chromosome.shape)
        chromosome[mutation_array] = mutated_values[mutation_array]
    return model_genetic
