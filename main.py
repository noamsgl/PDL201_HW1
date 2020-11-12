import numpy as np
from scipy.io import loadmat

x = loadmat('NNdata/PeaksData.mat')
Ct, Cv, Yt, Yv = x['Ct'], x['Cv'], x['Yt'], x['Yv']


# Write code for computing the objective loss function “soft-max” and its gradient with
# respect to wj and the biases. Make sure that the derivatives are correct using the
# gradient test. You should submit the results of the gradient test.


def softmax(W, X, C):
    """
    l = [num of categories]
    m = [num of samples]
    d = [num of features per sample]
    :param W: array of l weight vectors of dimension d = [shape = (d,l)]
    :param X: array of m samples, of dimension d. [shape = (d,m)]
    :param C: array of m categorical labels [shape = (l, m)]
    :return: objective loss function “soft-max”
    """
    l = len(Ct)
    m = len(X.T)
    d = len(W)
    eta = max([np.dot(X.T, W[j]) for j in range(d)])

    return -np.sum([C[k].T * np.log(np.exp(np.dot(X.T, W[k]) - eta) / np.exp(X.T @ W - eta)) for k in range(l)]) / m



