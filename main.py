import numpy as np
from scipy.io import loadmat

# Initialize
filenames = ['NNdata/GMMData.mat',
             'NNdata/PeaksData.mat',
             'NNdata/SwissRollData.mat']
x = loadmat(filenames[2])
Ct, Cv, Yt, Yv = x['Ct'], x['Cv'], x['Yt'], x['Yv']


# Question 1
def softmax(W, X, C, b):
    """
    l = [num of categories]
    m = [num of samples]
    d = [length of each sample]
    :param W: array[shape=(d,l)], each column is a weight vector of dimension d
    :param X: array[shape=(d,m)], each column is a data sample of dimension d
    :param C: array[shape=(l,m)], each column is a one-hot vector categorical label
    :param b: array[shape=(

    :return: objective loss function “soft-max”
    """

    d = len(W)
    m = len(X.T)
    l = len(C)
    eta = max([np.dot(X.T, W[j]) for j in range(d)])

    return -np.sum([C[k].T * np.log(np.exp(np.dot(X.T, W[k]) - eta) / np.exp(X.T @ W - eta)) for k in range(l)]) / m


def dsoftmax_db():
    pass


def compute_soft_max_gradient():
    """compute the gradient of the softmax function with respect to Wj and the bias.
    See question 1.
    """
    raise NotImplementedError


def test_derivates():
    """
    Makes sures the derivatives are correct using the gradient test.
    See question 1
    :return: results of the gradient test
    """
    raise NotImplementedError



def gradient_test():
    """make sure that the derivatives are okay using the gradient test"""
    raise NotImplementedError


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
