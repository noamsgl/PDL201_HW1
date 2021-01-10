import numpy as np
from matplotlib import pyplot as plt


def get_mini_batches(X, C, mini_batch_size):
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


def affine_transform(W, X, B):
    """
    :param W: (dim_output, dim_input)
    :param X: (dim_input,  num_sample)
    :param B: (dim_output)
    :return: (dim_output, num_sample) W@X+B per sample
    """
    return np.tensordot(W, X, axes=([1], [0])) + B


# softmax function
def softmax(Z):
    """
    :param Z : (dim_output, num_samples) values to apply softmax on
    :return : (dim_output, num_samples) softmax(Z) on every column
    """
    max_z = np.max(Z)
    exps = np.exp(Z - max_z)
    return exps / np.sum(exps, axis=0)


def F_objective(A, C):
    """
    cross entropy loss objective function.
    :param A:  (dim_output [, samples]) activation vector after softmax
    :param C:  (dim_output, samples) classes matrix as one hot vector for each sample
    :return: loss as scalar
    """
    num_samples = C.shape[1]
    a = np.log(A)  # a is (dim_output, [, samples])
    # loss = np.sum(a * C)  # loss for each class, loss is (1, dim_output)
    return (-1 / num_samples) * np.sum(a * C)


def initialize_data_for_test(input_dim, m, output_dim, L):
    x = np.random.randn(input_dim, m)
    c = np.zeros((output_dim, m))
    labels = np.random.choice(range(output_dim), size=m)
    # set c to be '1-hot' vectors for each label, shape [output_dim, m]
    c[labels, np.arange(m)] = 1

    return c, x


def initialize_theta(layers_sizes, init_with_none=False):
    theta = {}
    for i, (d1, d2) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:]), 1):
        theta[i] = {}
        theta[i]['w'] = None if init_with_none else np.random.random((d2, d1))
        theta[i]['b'] = None if init_with_none else np.random.random((d2, 1))
    return theta


def initialize_steps(layers_sizes):
    theta = {}
    for i, (d1, d2) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:]), 1):
        theta[i] = {}
        theta[i]['w'] = np.zeros((d2, d1))
        theta[i]['b'] = np.zeros((d2, 1))
    return theta


def grad_F(X, A, C):
    """
    :param X:  (dim_input, num_samples) input samples
    :param A:  (dim_output, num_samples) activation vector after softmax
    :param C:  (dim_output, num_samples) classes matrix as one hot vector for each sample
    :return: (l, n) gradient for each param
    """
    m = C.shape[1]
    return (1 / m) * (A - C) @ X.T


def plot_gradient_test_old(num_samples, input_dimension, output_dimension, iterations):
    """
    :param num_samples: number of samples
    :param input_dimension:
    :param output_dimension:
    :param iterations:
    :return: plots the results of the gradient test
    """

    # initialize random variables:
    X = np.random.rand(input_dimension - 1, num_samples)
    W = np.random.rand(output_dimension, input_dimension - 1)
    B = np.random.rand(output_dimension, 1)
    y = np.random.choice(range(output_dimension), size=num_samples)
    C = np.zeros((output_dimension, num_samples))
    C[y, np.arange(num_samples)] = 1
    D = np.random.rand(output_dimension, input_dimension - 1)

    # initialize results lists:
    val1_lst = [0]
    val2_lst = [0]
    ratio1_list = []
    ratio2_list = []

    # iterate:
    for i in range(iterations):
        # compute values:
        epsilon = (0.5 ** i)
        A0 = softmax(affine_transform(W, X, B))
        A1 = softmax(affine_transform(W + epsilon * D, X, B))
        f0 = F_objective(A0, C)
        f1 = F_objective(A1, C)
        dh_grad = epsilon * np.sum(grad_F(X, A0, C) * D)
        val1 = np.linalg.norm(f1 - f0)
        val2 = np.linalg.norm(f1 - f0 - dh_grad)
        # archive results:
        ratio1_list.append(val1_lst[-1] / val1)
        ratio2_list.append(val2_lst[-1] / val2)
        val1_lst.append(val1)
        val2_lst.append(val2)

    # plot results:
    X_axis = np.arange(iterations)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('The Gradient Test', fontsize='x-large', y=1.1)
    axs[0].set_title(r'$val_1, val_2$ on a linear scale')
    axs[0].set_xlabel('iteration')
    axs[0].plot(X_axis, val1_lst[1:], label=r'$val_1$')
    axs[0].plot(X_axis, val2_lst[1:], label=r'$val_2$')
    axs[0].legend()
    axs[0].set_facecolor('w')
    axs[1].set_title(r'$val_1, val_2$ on a semilog scale')
    axs[1].set_xlabel('iteration')
    axs[1].semilogy(X_axis, val1_lst[1:], label=r'$val_1$')
    axs[1].semilogy(X_axis, val2_lst[1:], label=r'$val_2$')
    axs[1].legend()
    axs[1].set_facecolor('w')
    axs[2].set_title('Ratio Test')
    axs[2].set_xlabel('iteration')
    axs[2].plot(X_axis, ratio1_list, label=r'$\frac{\epsilon_i}{val_1}$')
    axs[2].plot(X_axis, ratio2_list, label=r'$\frac{\epsilon_i^2}{val_2}$')
    axs[2].set_ylim([0, 10])
    axs[2].legend()
    axs[2].set_facecolor('w')
    plt.show()
