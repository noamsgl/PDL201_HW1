from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from tqdm import tqdm

from network import Network

DATASETS_PATH = ['NNdata/SwissRollData.mat',
                 'NNdata/PeaksData.mat',
                 'NNdata/GMMData.mat']

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def run_part_3(learning_rates=(0.1, 0.3), batch_sizes=(32, 64), hidden_layers_for_dataset=None, iters=50,
               optimizer=None, gamma=0.7):
    n_learning_rates = len(learning_rates)
    n_batch_sizes = len(batch_sizes)
    for filename in DATASETS_PATH:
        if hidden_layers_for_dataset is None:
            hidden_layers = []
        else:
            hidden_layers = hidden_layers_for_dataset[filename]
        x = loadmat(filename)
        Xt, Xv, Ct, Cv, = x['Yt'], x['Yv'], x['Ct'], x['Cv']
        Y = np.argmax(Ct, axis=0)
        datasets = {'train': (Xt, Ct),
                    'test': (Xv, Cv)}

        sns.scatterplot(x=Xt[0], y=Xt[1], hue=Y, palette='pastel', linewidth=0).set_title("2d visualization of {}".format(filename))

        fig = plt.figure(figsize=(6 * n_batch_sizes, 6 * n_learning_rates))
        fig.suptitle("Accuracy vs. Epoch\nDataset: {}, Network Layers: {}".format(filename, hidden_layers), fontsize=50)

        for i, (lr, batch_size) in tqdm(enumerate(list(product(learning_rates, batch_sizes))), desc=filename,
                                        total=n_batch_sizes * n_learning_rates):
            input_dimension = Xt.shape[0]
            output_dimension = Ct.shape[0]
            net = Network(input_dimension, output_dimension, hidden_layers_sizes=hidden_layers, learning_rate=lr,
                          optimizer=optimizer, gamma=gamma)
            x_plot, y_plots = net.train(Xt, Ct, test_data=datasets, max_epochs=iters, batch_size=batch_size)
            ax = fig.add_subplot(len(learning_rates), len(batch_sizes), i + 1)
            ax.set_title(r"$\alpha={}, bs={}$".format(lr, batch_size))
            ax.set_ylim([0, 1])
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            for ds_name, ds_accuracies in y_plots.items():
                ax.plot(ds_accuracies, label=ds_name)
            ax.legend()
        plt.show()


if __name__ == '__main__':
    # HIDDEN_LAYERS_FOR_DATASET = {
    #     'NNdata/SwissRollData.mat': [4, 6, 8],
    #     'NNdata/PeaksData.mat': [4, 6, 8, 10, 10, 10],
    #     'NNdata/GMMData.mat': [6, 7, 9, 11, 11, 10]
    # }

    HIDDEN_LAYERS_FOR_DATASET = {
        'NNdata/SwissRollData.mat': [4, 8, 8, 5],
        'NNdata/PeaksData.mat': [6, 8, 10],
        'NNdata/GMMData.mat': [6, 8, 10]
    }
    # run_part_3(with_hidden_layers=False, optimizer=None)
    # run_part_3(with_hidden_layers=False, optimizer='momentum', gamma=0.7)
    run_part_3(hidden_layers_for_dataset=HIDDEN_LAYERS_FOR_DATASET, optimizer=None)
    run_part_3(hidden_layers_for_dataset=HIDDEN_LAYERS_FOR_DATASET, optimizer='momentum', gamma=0.7)
