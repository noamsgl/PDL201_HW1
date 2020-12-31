import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Initialize
filenames = ['NNdata/GMMData.mat',
             'NNdata/PeaksData.mat',
             'NNdata/SwissRollData.mat']
x = loadmat(filenames[2])
Ct, Cv, Yt, Yv = x['Ct'], x['Cv'], x['Yt'], x['Yv']


def objective(X, C, weights, biases):
    """
    gets weights and biases of layer L and returns cross-entropy loss
    :param weights: weight matrix
    :param biases: biases vector
    :param X: output from L-1
    :param Y: label for this sample
    :return:
    """
    z = weights @ X.T + biases
    return C @ (-np.log(softmax(z)))


def softmax(vec):
    """
    :param vec:  [shape = (d,1)] vec
    :return: softmax(vec)
    """
    eta = max(vec)
    vec = vec - eta
    sum = np.sum([np.exp(x) for x in vec])
    return np.exp(vec) / sum


def grad_objective_w(X, C, W, b, p=0):
    m = len(X.T)
    return (1 / m) * X @ (np.exp(X.T @ W[p] + b) / (np.sum([np.exp(W[j].T @ X + b) for j in range(len(W))])) - C)


def grad_objective_b():
    pass


def gradient_test():
    # initialize
    X_plot, Y_1, Y_2 = [], [], []
    np.random.seed(18)
    dimension = 10

    X = np.random.randint(0, 11, size=dimension)
    C = np.zeros(shape=dimension)
    C[0] += 1
    weights = np.random.randint(0, 11, size=(dimension, dimension))
    biases = np.random.randint(0, 11, size=dimension)
    d = np.random.random(size=dimension)

    objective(X, C, weights, biases)

    epsilon = 1

    for i in range(100):
        X_plot.append(i)
        epsilon = (1 / 2) ** i * epsilon
        val1 = abs(objective(X, C, weights + epsilon*d, biases) - objective(X, C, weights, biases))
        val2 = abs(objective(X, C, weights + epsilon*d, biases) - objective(X, C, weights, biases) - epsilon * d.T @ grad_objective_w(X, C, weights, biases))
        Y_1.append(val1)
        Y_2.append(val2)
    plt.plot(X_plot, Y_1, label='val1')
    plt.plot(X_plot, Y_2, label='val2')
    plt.show()


# Question 2
def minimize_objective_function_SGD(func):
    """
    Minimizes an objective function.
    See question 2.
    :param func:
    :return: optimal params
    """
    raise NotImplementedError


# Question 3
def demonstrate_minimization_of_softmax():
    """
    Plots a graph of accuracy of data classification after each epoch.
    Do the plots for both the training data and the validation data.
    See question 3.
    :return:
    """
    raise NotImplementedError


# Question 4
def code_for_neural_network():
    """
    Includes the forward pass and back-ward pass (the computation of the "Jacobian transpose times vector").
    See question 4.
    :return:
    """
    raise NotImplementedError


def test_jacobians():
    """
    See that the Jacobian tests work and submit the tests.
    See question 4
    :return:
    """


# Question 6
def compute_forward_pass_on_network(L):
    """
    Compute a forward pass of a network with L layers
    :param L: num of layers
    :return:
    """


# Question 6
def compute_backward_pass_on_network(L):
    """
    Compute a backward pass of a network with L layers
    :param L: num of layers
    :return:
    """


def question_6(L):
    """
    Compute a forward pass.
    Compute a backward pass.
    See the gradient of the whole network passes the gradient test
    :param L: num of layers
    :return gradient test
    """
    compute_forward_pass_on_network(L)
    compute_backward_pass_on_network(L)
    gradient_test()


# Recycle Bin

def grad_objective_data(X, C, W, b):
    """
    See section 1.3.3
    l = num of labels
    :param X: [x_1 | x_2 | ... | x_m ] in R^(l * m)
    :param C: [c_1 | c_2 | ... | c_m ] in R^(l * m)
    :param W: [w_1 | w_2 | ... | w_l ] in R^(n * l)
    :param b: [biases] in R^l
    :return: gradient of objective function w.r.t. data
    """
    m = len(X.T)
    return (1 / m) * W @ (np.exp(W.T @ X + b) / (np.sum([np.exp(W[j].T @ X + b) for j in range(len(W))])) - C)
