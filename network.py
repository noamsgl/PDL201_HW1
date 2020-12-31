import numpy as np
from utils import *


class Network:
    def __init__(self, layers, alpha=0.15):
        """
        Instantiate a new neural network.
        :param layers: network architecture, e.g. [2, 4, 4, 2]
        """
        self.layers = layers
        self.L = len(self.layers)
        self.weights = self.initialize_weights(layers)
        self.biases = self.initialize_biases(layers)
        self.alpha = alpha
        self.loss =

    def stochastic_gradient_descent(self, data, num_epochs, mini_batch_size, test_data=None):
        for epoch in range(num_epochs):
            mini_batches = self.get_mini_batches(data, mini_batch_size)
            for mini_batch in mini_batches:

                for x, y in mini_batch:
                    gradients = self.backpropagation(x, y)

                self.train(mini_batch)

    def train(self, mini_batch):
        pass

    @staticmethod
    def initialize_weights(layers):
        list_of_weights = [np.random.random((size_out, size_in)) for size_out, size_in in zip(layers[:-1], layers[1:])]
        return np.dstack(list_of_weights)

    @staticmethod
    def initialize_biases(layers):
        list_of_biases = [np.random.random(size_in) for size_in in layers[1:]]
        return np.dstack(list_of_biases)

    @staticmethod
    def get_mini_batches(self, data, mini_batch_size):
        np.random.shuffle(data)
        mini_batches = [data[i:i + mini_batch_size] for i in range(0, len(data), mini_batch_size)]
        return mini_batches

    def feedforward(self, mini_batch):
        pass

    def backpropagation(self, x, y):
        pass

    def update_weights(self, gradients):
        self.weights -= self.alpha * () / m

    def update_biases(self, gradients):
        self.biases
