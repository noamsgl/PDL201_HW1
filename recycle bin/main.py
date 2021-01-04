from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from network import Network

DATASETS_PATH = ['NNdata/SwissRollData.mat',
                 'NNdata/PeaksData.mat',
                 'NNdata/GMMData.mat',
                 ]

# gradient_test(m=50, n=3, l=5, iterations=20)

for filename in DATASETS_PATH:
    x = loadmat(filename)
    Xt, Xv, Ct, Cv, = x['Yt'], x['Yv'], x['Ct'], x['Cv']
    datasets = {'train': (Xt, Ct),
                'test': (Xv, Cv)}
    n_learning_rates = 4
    n_batch_sizes = 4
    learning_rates = np.logspace(-1, -4, n_learning_rates, base=10)
    batch_sizes = np.logspace(4, 7, n_batch_sizes, base=2, dtype=int)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Accuracy vs. Epoch on {} dataset".format(filename), fontsize=20)
    for i, (lr, batch_size) in tqdm(enumerate(list(product(learning_rates, batch_sizes))), desc=filename,
                                    total=n_batch_sizes * n_learning_rates):
        input_dimension = Xt.shape[0]
        output_dimension = Ct.shape[0]
        net = Network(input_dimension, output_dimension, L=0)
        x_plot, y_plots = net.stochastic_gradient_descent(Xt, Ct, lr, batch_size, test_data=datasets)
        ax = fig.add_subplot(len(learning_rates), len(batch_sizes), i + 1)
        ax.set_title(r"$\alpha={}, bs={}$".format(lr, batch_size), fontsize=16)
        for ds_name, ds_accuracies in y_plots.items():
            ax.plot(ds_accuracies, label=ds_name)
        ax.legend()
    plt.show()
