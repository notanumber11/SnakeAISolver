import os
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


class DeepNeuralNetModel:
    MODEL_FEATURE_COUNT = 5
    MODEL_NAME = "Basic"
    hidden = None
    hidden_node_neurons = MODEL_FEATURE_COUNT ** 3

    def __init__(self, path):
        self.dnn_model_path = path
        self.dnn_model_file_name = self.dnn_model_path + DeepNeuralNetModel.MODEL_NAME
        network = input_data(shape=[None, DeepNeuralNetModel.MODEL_FEATURE_COUNT, 1])
        self.hidden = network = fully_connected(network, self.hidden_node_neurons, activation='relu6')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', loss='mean_square')
        self.model = tflearn.DNN(network)
        # if os.path.isfile(self.dnn_model_file_name+".index"):
        #     self.model.load(self.dnn_model_file_name)

    def save(self):
        self.model.save(self.dnn_model_file_name)

    def get_weights(self):
        return self.model.get_weights(self.hidden.W)

    def set_weights(self, weights):
        self.model.set_weights(self.hidden.W, weights)
