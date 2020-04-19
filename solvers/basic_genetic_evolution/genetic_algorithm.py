from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from solvers.basic_dnn import constants, training_data_generator
import numpy as np


class GeneticAlgorithm:

    def __init__(self, layers_size: List[int]):
        self.layers_size = layers_size
        self.number_of_layers = len(self.layers_size)

    def get_initial_population_genetic(self, population_size: int) -> List:
        """
        Each element in the population is a neural network
        with the dimensions defined in self.layers_size.
        For each layer we need to initialize the weights.
        The weights of each layer have dimensionality: [size of previous layer, size of current layer].
        Notice that we are not tuning the biases of each layer.
        :param population_size:
        :return: List of population size containing model_genetics. Each model_genetic is a list
        of weights for each layer in the neural network.
        """
        population_genetic = []
        for i in range(population_size):
            # For each layer we need to create the original parameters
            model_genetic = []
            for j in range(1, self.number_of_layers):
                rows = self.layers_size[j - 1]
                columns = self.layers_size[j]
                weights = np.random.uniform(low=-1, high=1, size=(rows, columns))
                model_genetic.append(weights)
            population_genetic.append(model_genetic)
        return population_genetic

    def evaluate_population(self, population_genetic, game_status):
        model = self.build_model()
        weights = model.get_weights()
        population_evaluated = []
        for model_genetic in population_genetic:
            for i in range(0, len(weights), 2):
                if weights[i].shape != model_genetic[i//2].shape:
                    raise ValueError("Error setting model shapes")
                weights[i] = model_genetic[i//2]
            model.set_weights(weights)
            inputs = training_data_generator.get_input_from_game_status(game_status)

            print("-----------------------------------------------")
            result = model.predict([[1, 0, 0, 0, 1, 1, 0, 0, 0.8]])
            print(result)
            print("-----------------------------------------------")

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(self.layers_size[1], activation='relu', input_shape=[self.layers_size[0]]),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        # model.summary()
        return model
