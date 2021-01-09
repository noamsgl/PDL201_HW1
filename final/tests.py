from network import Network
import utils
import numpy as np
from matplotlib import pyplot as plt


def gradient_test_loss_for_param(input_dim, output_dim, jacobian=False, samples=1,
                                 layer=1, param='w', hidden_layers_sizes=None, iters=15, eps0=1, optimizer=None):
    net = Network(input_dim, output_dim, hidden_layers_sizes=hidden_layers_sizes, optimizer=optimizer)
    c, x = utils.initialize_data_for_test(input_dim, samples, output_dim, net.L)
    delta = utils.initialize_theta(net.layers_sizes)
    eps_diff = [0]
    ratio_eps = []
    eps_sqr_diff = [0]
    ratio_eps_sqr = []
    epsilon = eps0
    for i in range(iters):
        # epsilon = (0.5 ** i) * eps0
        fx, _ = net.forward(x, c)
        if jacobian:
            fx = net.a[layer]
            df_dw, _, df_db = net.back_hidden(samples, layer, np.ones((net.layers_sizes[layer], samples)))
            grad = df_dw if param == 'w' else df_db
        else:
            net.backprop(c)
            grad = net.get_grads(layer, param)
        eps_delta = epsilon * delta[layer][param]
        net.set_theta_for_grad_test(layer, param, eps_delta)

        fx_ed, _ = net.forward(x, c)
        if jacobian:
            fx_ed = net.a[layer]
            eps_delta_grad_x = np.sum((eps_delta * grad), axis=1, keepdims=True)
        else:
            eps_delta_grad_x = (eps_delta.reshape(1, -1) @ grad.reshape(-1, 1))[0, 0]
        eps_diff.append(np.linalg.norm(fx_ed - fx))
        ratio_eps.append(eps_diff[-2] / eps_diff[-1])
        eps_sqr_diff.append(np.linalg.norm(fx_ed - fx - eps_delta_grad_x))
        ratio_eps_sqr.append(eps_sqr_diff[-2] / eps_sqr_diff[-1])

        epsilon = 0.5 * epsilon
    plot_gradient_test(layer, param, ratio_eps, ratio_eps_sqr, iters)


# this test with no hidden layers tests the loss grad test
def gradient_test_for_all_params(input_dim, output_dim, jacobian=False, samples=1, hidden_layers_sizes=None, iters=15,
                                 eps0=1, optimizer=None):
    L = 1 if hidden_layers_sizes is None else (len(hidden_layers_sizes) + 1)
    for l in range(1, L + 1):
        gradient_test_loss_for_param(input_dim, output_dim, iters=iters, jacobian=jacobian, samples=samples,
                                     hidden_layers_sizes=hidden_layers_sizes, layer=l, param='w', eps0=eps0, optimizer=optimizer)
        gradient_test_loss_for_param(input_dim, output_dim, jacobian=jacobian, samples=samples,
                                     hidden_layers_sizes=hidden_layers_sizes, layer=l, param='b', eps0=eps0, optimizer=optimizer)


def gradient_test_for_hidden_layer_jacobian(input_dim, output_dim, jacobian=True, samples=1, hidden_layers_sizes=None, iters=15,
                                 eps0=1, optimizer=None):
    hidden_layers = [int(np.ceil((input_dim + output_dim) / 2))] if hidden_layers_sizes is None else hidden_layers_sizes
    L = (len(hidden_layers) + 1)
    for l in range(1, L):
        gradient_test_loss_for_param(input_dim, output_dim, jacobian=jacobian, samples=samples,
                                     layer=l, param='w', hidden_layers_sizes=hidden_layers, iters=iters, eps0=eps0, optimizer=optimizer)
        gradient_test_loss_for_param(input_dim, output_dim, jacobian=jacobian, samples=samples,
                                     layer=l, param='b', hidden_layers_sizes=hidden_layers, iters=iters, eps0=eps0, optimizer=optimizer)


def plot_gradient_test(layer, param, ratio1_list, ratio2_list, iterations):
    # plot results:
    X_axis = np.arange(iterations)
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    fig.suptitle('The Gradient Test', fontsize='x-large', y=1.1)
    axs.set_title(f'Ratio Test for {param}_{layer}')
    axs.set_xlabel('iteration')
    axs.plot(X_axis, ratio1_list, label=r'$\frac{\epsilon_i}{val_1}$')
    axs.plot(X_axis, ratio2_list, label=r'$\frac{\epsilon_i^2}{val_2}$')
    axs.set_ylim([0, 10])
    axs.legend()
    axs.set_facecolor('w')
    plt.show()


hidden_layers_for_tests = [4, 5, 5]
input_dim = 4
output_dim = 4
# network with no hidden layers - equivalent to test the loss layer
gradient_test_for_all_params(input_dim, output_dim, samples=1, jacobian=False, hidden_layers_sizes=None, iters=15, optimizer='None')
# network with hidden layers - test every param in the network
gradient_test_for_all_params(input_dim, output_dim, samples=1, jacobian=False, hidden_layers_sizes=hidden_layers_for_tests, iters=15, optimizer='None')
# jacobian test for a hidden layer derivative - when given no layers (None), create one layer and checks it
gradient_test_for_hidden_layer_jacobian(input_dim, output_dim, samples=1, jacobian=True, hidden_layers_sizes=None, iters=15, optimizer='None')
# jacobian test for a hidden layer derivative - checks every parameter in the hidden layers
gradient_test_for_hidden_layer_jacobian(input_dim, output_dim, samples=1, jacobian=True, hidden_layers_sizes=hidden_layers_for_tests, iters=15, optimizer='None')

