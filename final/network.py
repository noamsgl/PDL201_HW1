from collections import defaultdict
import numpy as np
from utils import affine_transform, softmax, F_objective, initialize_theta, get_mini_batches, initialize_steps

TANH = "tanh"
RELU = "relu"
SOFTMAX = "softmax"


class Network:

    def __init__(self, input_dimension, output_dimension, hidden_layers_sizes=None,
                 activation=TANH, loss_func=SOFTMAX, optimizer=None, learning_rate=0.1, gamma=0.5):
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
        self.theta = initialize_theta(self.layers_sizes) # dict from layerNum - l to (weights_l, bias_l)
        self.z = {} # dict from layerNum - l to [w_l @ x + b_l], shape - [layers_sizes[l], layers_sizes[l-1]]
        self.a = {} # dict from layerNum - l to activation(z), shape - [layers_sizes[l], layers_sizes[l-1]]
        self.activation = activation
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optimizer
        if optimizer:
            self.steps_for_opt = initialize_steps(self.layers_sizes)
        self.w_grads = {} # dict layerNum -> grad of theta[l]['w'] to remember calculations of derivatives for backpropogation
        self.b_grads = {} # dict layerNum -> grad of theta[l]['b'] to remember calculations of derivatives for backpropogation

        self.theta_for_grad_test = initialize_theta(self.layers_sizes, init_with_none=True)

    def set_theta_for_grad_test(self, l, p, theta_for_test):
        self.theta_for_grad_test[l][p] = theta_for_test.copy()

    def get_w_b(self, l):
        w, b = self.theta[l]['w'], self.theta[l]['b']
        if self.theta_for_grad_test[l].get('w') is not None:
            w += self.theta_for_grad_test[l].get('w')
            self.theta_for_grad_test[l].clear()
        if self.theta_for_grad_test[l].get('b') is not None:
            b += self.theta_for_grad_test[l].get('b')
            self.theta_for_grad_test[l].clear()
        return w, b

    def get_grads(self, l, p):
        if p == 'w':
            return self.w_grads[l]
        else:
            return self.b_grads[l]

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
        return F_objective(out, c), out

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

    def update_theta(self, c):
        for l in range(1, self.L + 1):
            if self.optimizer is None:
                w_dec = self.w_grads[l] * self.learning_rate
                b_dec = self.b_grads[l] * self.learning_rate
            elif self.optimizer == "momentum":
                self.steps_for_opt[l]['w'] = self.steps_for_opt[l]['w'] * self.gamma + self.w_grads[l] * self.learning_rate
                w_dec = self.steps_for_opt[l]['w']
                self.steps_for_opt[l]['b'] = self.steps_for_opt[l]['b'] * self.gamma + self.b_grads[l] * self.learning_rate
                b_dec = self.steps_for_opt[l]['b']
            # TODO - check if clip func is good for us, and if needed to be done on each vector seperatly
            #  or can be done on all matrix
            self.theta[l]['w'] = np.clip(self.theta[l]['w'] - w_dec, -1, 1)
            self.theta[l]['b'] = np.clip(self.theta[l]['b'] - b_dec, -1, 1)
            # self.theta[l]['w'] = self.theta[l]['w'] - w_dec
            # self.theta[l]['b'] = self.theta[l]['b'] - b_dec

    def train(self, Xt, Ct, test_data=None, max_epochs=100, batch_size=100):
        assert Xt.shape[0] == self.input_dim, 'input dimension of Xt is different from NET input_dim'
        assert Xt.shape[1] == Ct.shape[1], 'number of samples is different between Xt and Ct'
        if test_data:
            epoch_num_for_plot, accuracy_for_epoch = [], defaultdict(list)

        for epoch in range(1, max_epochs+1):
            if test_data:
                epoch_num_for_plot.append(epoch)
                for ds_name, ds in test_data.items():
                    accuracy_for_epoch[ds_name].append(self.score(ds))

            for x, c in get_mini_batches(Xt, Ct, batch_size):
                loss, _ = self.forward(x, c)
                self.backprop(c)
                self.update_theta(c)

        if test_data:
            return epoch_num_for_plot, accuracy_for_epoch

    def predict(self, X, c):
        """
        Return the most likely label for x
        :param X: (input_dimension, num_samples)
        :return:
        """
        # A = softmax(affine_transform(self.weights, X, self.biases))
        # a.shape [output_dim, samples]
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
