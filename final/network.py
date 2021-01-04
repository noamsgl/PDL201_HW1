from collections import defaultdict

import numpy as np

from utils import affine_transform, softmax, grad_F


class Network:

    def __init__(self, input_dimension, output_dimension, L=2, hidden_dimension=16):
        """
        Instantiate a new neural network.
        :param input_dimension:
        :param output_dimension:
        :param L: number of hidden layers
        :param hidden_dimension:
        """

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.L = L
        self.hidden_dimension = hidden_dimension
        self.weights = self.initialize_weights(L)
        self.biases = self.initialize_biases(L)

    def stochastic_gradient_descent(self, X, C, learning_rate=0.15, mini_batch_size=100, num_epochs=100, epsilon=1e-4,
                                    test_data=None):
        """
        Optimize W for the given X,Y using stochastic gradient descent
        :param X: (input_dimension, num_samples)
        :param C: (output_dimension, num_samples)
        :param learning_rate:
        :param num_epochs:
        :param mini_batch_size:
        :param test_data: dict<name, test set>. if given, the SGD will evaluate on the given test_data and return results.
        :return: x_plot, y_plots: values for plotting results
        """

        n, m = X.shape
        l, _ = C.shape
        Ws = self.weights.flatten()
        x_plot, y_plots = [], defaultdict(list)

        # loop num_epochs times
        for i in range(1, num_epochs):
            # iterate over mini-batches
            for X, C in self.get_mini_batches(X, C, mini_batch_size):
                A = softmax(affine_transform(self.weights, X, self.biases))
                gradient = grad_F(X, A, C)
                self.weights = self.weights - learning_rate * gradient
            #     todo: update biases

            # Ws (num_epochs, output_dimension*input_dimension) - a flattened weights array for each epoch
            Ws = np.r_[Ws, self.weights.flatten()]

            if test_data is not None:
                x_plot.append(i)
                for ds_name, ds in test_data.items():
                    y_plots[ds_name].append(self.score(ds))

            if np.linalg.norm(Ws[-1] - Ws[-2]) / np.linalg.norm(Ws[-2]) < epsilon:
                break

        return x_plot, y_plots

    def predict(self, X):
        """
        Return the most likely label for x
        :param X: (input_dimension, num_samples)
        :return:
        """
        A = softmax(affine_transform(self.weights, X, self.biases))
        return np.argmax(A, axis=0)

    def score(self, dataset):
        """
        :param datasets: dataset to check score for
        :return: dict of scores
        """
        X, C = dataset
        Y_true = np.argmax(C, axis=0)
        Y_predicted = self.predict(X)
        accuracy = np.mean(Y_predicted == Y_true)
        return accuracy

    def get_mini_batches(self, X, C, mini_batch_size):
        """
        A generator which yields mini_batches
        :param X:
        :param C:
        :param mini_batch_size:
        :return:
        """
        n, m = X.shape
        l, _ = C.shape
        num_batches = int(m / mini_batch_size)

        batches_idxs = np.array_split(np.random.permutation(m), num_batches)
        for batch_idxs in batches_idxs:
            mbatch = batch_idxs.size
            yield X[:, batch_idxs].reshape(n, mbatch), C[:, batch_idxs].reshape(l, mbatch)

    def initialize_weights(self, L):
        # todo: implement
        list_of_weights = []
        # 1. Add input layer weights
        # 2. Add hidden layers
        # 3. Add output layer
        if L == 0:
            return np.random.random((self.output_dimension, self.input_dimension))

    def initialize_biases(self, L):
        # todo: implement
        list_of_biases = []
        # 1. Add input layer biases
        # 2. Add hidden layers
        # 3. Add output layer
        if L == 0:
            return np.random.random((self.output_dimension, 1))
