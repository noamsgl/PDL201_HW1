from collections import defaultdict

import numpy as np

from utils import affine_transform, softmax, grad_F, F_objective

TANH = "tanh"
RELU = "relu"
SOFTMAX = "softmax"


class Network:

    def __init__(self, input_dimension, output_dimension, hidden_layers_sizes=None, activation=TANH, loss_func=SOFTMAX):
        """
        Instantiate a new neural network.
        :param input_dimension:
        :param output_dimension:
        :param hidden_layers_sizes: list of sizes of the hidden layers, None will build 1 layer from input to output
        """
        if hidden_layers_sizes is None:
            hidden_layers_sizes = []
        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.L = len(hidden_layers_sizes) + 1
        self.layers_sizes = [input_dimension] + hidden_layers_sizes + [output_dimension]
        self.theta = self.initialize_theta() # dict from layerNum - l to (weights_l, bias_l)
        self.z = {} # dict from layerNum - l to [w_l @ x + b_l], shape - [layers_sizes[l], layers_sizes[l-1]]
        self.a = {} # dict from layerNum - l to activation(z), shape - [layers_sizes[l], layers_sizes[l-1]]
        self.activation = activation
        self.loss_func = loss_func

        self.w_grads = {} # dict to remember calculations of derivatives for backpropogation
        self.b_grads = {} # dict to remember calculations of derivatives for backpropogation

    def get_w_b(self, l):
        return self.theta[l]['w'], self.theta[l]['b']

    def forward(self, x, c):
        """
        get a sample or batch, return loss and save data
        :param x: samples, [input_dim, samples]
        :param c: classes as '1 hot' vectors, [output_dim, samples]
        :return: loss as scalar
        """
        # if x.shape[1] != c.shape[1]:
        assert x.shape[1] == c.shape[1], "Number of samples in x is not equals to num of samples in c, check dims"
        self.z[0] = x.copy()
        self.a[0] = x.copy()
        for l in range(1, self.L):
            self.hidden_forward(l)
        out = self.last_forward()
        return F_objective(self.a[self.L], c), out

    def last_forward(self):
        w, b = self.get_w_b(self.L)
        self.z[self.L] = affine_transform(w, self.a[self.L - 1], b)
        self.a[self.L] = softmax(self.z[self.L])
        return self.a[self.L]

    def hidden_forward(self, l):
        w, b = self.get_w_b(l)
        self.z[l] = affine_transform(w, self.a[l - 1], b)
        if self.activation == TANH:
            self.a[l] = np.tanh(self.z[l])
        elif self.activation == RELU:
            self.a[l] = np.maximum(0, self.z[l])

    def back_objective(self, c, m):
        """
        start the backprop process, gradient on the last layer w.r.t c
        :param m: samples
        :param c: the labels
        :return: df_dw, df_dx, df_db
        """

        # [output_dim, m]
        df_dz = (1/m) * (self.a[self.L] - c)
        # [self.layers_sizes[self.L - 1], m]
        dz_dw = self.a[self.L - 1]
        # [output_dim = self.layers_sizes[self.L], self.layers_sizes[self.L - 1]]
        dz_dx, _ = self.get_w_b(self.L)
        # [output_dim = self.layers_sizes[self.L], self.layers_sizes[self.L - 1]]
        df_dw = df_dz @ dz_dw.T
        # [self.layers_sizes[self.L - 1], m]
        df_dx = dz_dx.T @ df_dz
        # [output_dim, 1], dz_db is all ones
        df_db = df_dz @ np.ones((m, 1))

        return df_dw, df_dx, df_db

    def back_hidden(self, m, l, dF_da):
        """
        calc the derivative of a hidden layer
        :param m: num of samples in batch
        :param l: number of layer
        :param dF_da: the derivative of the l+1 layer w.r.t a_l, [self.layers_sizes[l], m]
                    each column in acc wil be multiply elementwise with df_dz columns instead of a diagonal matrix
        :return: df_dw, df_dx, df_db
        """

        # df_dz shape - [self.layers_sizes[l], m]
        if self.activation == TANH:
            df_dz = (1 - np.tanh(self.z[l]) ** 2) * dF_da
        elif self.activation == RELU:
            df_dz = np.where(self.z[l] > 0, 1, 0) * dF_da
        else:
            raise NotImplementedError("trying to backprop with a not familiar activation func")
        # [self.layers_sizes[l - 1], m]
        dz_dw = self.a[l-1]
        # [self.layers_sizes[l], self.layers_sizes[l - 1]]
        dz_dx, _ = self.get_w_b(l)
        # [self.layers_sizes[l], self.layers_sizes[l - 1]]
        # TODO - check if need (1/m) *
        df_dw = df_dz @ dz_dw.T
        # [self.layers_sizes[self.L - 1], m]
        df_dx = dz_dx.T @ df_dz
        # [output_dim, 1], dz_db is all ones
        # TODO - check if need (1/m) *
        df_db = df_dz @ np.ones((m, 1))

        return df_dw, df_dx, df_db

    def backprop(self, c):
        m = c.shape[1]
        self.w_grads[self.L], df_dx, self.b_grads[self.L] = self.back_objective(c, m)
        for l in range(self.L - 1, 0, -1):
            df_da = df_dx.copy()
            self.w_grads[l], df_dx, self.b_grads[l] = self.back_hidden(m, l, df_da)

    def optimize(self, c, optimizer=None, lr=0.1):
        self.backprop(c)
        if optimizer is None:
            for l in range(1, self.L + 1):
                self.theta[l]['w'] = self.theta[l]['w'] - self.w_grads[l] * lr
                self.theta[l]['b'] = self.theta[l]['b'] - self.b_grads[l] * lr
        # elif optimizer == "momentum" / "adagard":
            # todo some other optimizers

    def train(self, Xt, Ct, test_data=None, Xv=None, Cv=None, score_every_epoch=False, max_epochs=100, batch_size=100, learning_rate=0.1):
        assert Xt.shape[0] == self.input_dim, 'input dimension of Xt is different from NET input_dim'
        assert Xt.shape[1] == Ct.shape[1], 'number of samples is different between Xt and Ct'
        # assert (Xv is None and Cv is None) or (Xv is not None and Cv is not None), 'validation X and C must be similar'
        assert (test_data is None and not score_every_epoch) or (test_data and score_every_epoch)

        if score_every_epoch:
            x_plot, y_plots = [], defaultdict(list)
            # TODO - initialize dicts to save score for each epoch

        for epoch in range(1, max_epochs):
            for x, c in self.get_mini_batches(Xt, Ct, batch_size):
                loss, _ = self.forward(x, c)
                self.optimize(c, lr=learning_rate)
            if score_every_epoch:
                x_plot.append(epoch)
                for ds_name, ds in test_data.items():
                    y_plots[ds_name].append(self.score(ds))
                pass
                # TODO - save all scores verify
        if score_every_epoch:
            return x_plot, y_plots


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

    def predict(self, X, c):
        """
        Return the most likely label for x
        :param X: (input_dimension, num_samples)
        :return:
        """
        # A = softmax(affine_transform(self.weights, X, self.biases))
        _, A = self.forward(X, c)
        return np.argmax(A, axis=0)

    def score(self, dataset):
        """
        :param datasets: dataset to check score for
        :return: dict of scores
        """
        X, C = dataset
        Y_true = np.argmax(C, axis=0)
        Y_predicted = self.predict(X, C)
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

        # i = 1
        # for d1, d2 in zip(self.layers_sizes[:-1], self.layers_sizes[1:]):
        #     print(f'i={i}, d1={d1}, d2={d2}')
        #     theta[i] = {}
        #     theta[i]['w'] = np.random.random((d2, d1))
        #     theta[i]['b'] = np.random.random((d2, 1))
        #     i += 1
    def initialize_theta(self):
        theta = {}
        for i, (d1, d2) in enumerate(zip(self.layers_sizes[:-1], self.layers_sizes[1:]), 1):
            theta[i] = {}
            theta[i]['w'] = np.random.random((d2, d1))
            theta[i]['b'] = np.random.random((d2, 1))
        return theta

    def initialize_weights(self):
        weights = {}
        for i, d1, d2 in enumerate(zip(self.layers_sizes[:-1], self.layers_sizes[1:]), 1):
            weights[i] = np.random.random((d2, d1))
        return weights

    def initialize_biases(self):
        biases = {}
        for i, d2 in enumerate(self.layers_sizes[1:], 1):
            biases[i] = np.random.random((d2, 1))
        return biases
