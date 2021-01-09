from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import seaborn as sns
from network import Network
DATASETS_PATH = ['NNdata/SwissRollData.mat',
                 'NNdata/PeaksData.mat',
                 'NNdata/GMMData.mat']
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


def run_part_3(with_hidden_layers=False, iters=50, optimizer=None, gamma=0.7):
    learning_rates = [0.3, 0.1]
    n_learning_rates = len(learning_rates)
    batch_sizes = [32, 64]
    n_batch_sizes = len(batch_sizes)
    # learning_rates = np.logspace(-1, -4, n_learning_rates, base=10)
    # learning_rates = [0.3, 0.1, 0.01, 0.001]
    # batch_sizes = np.logspace(5, 8, n_batch_sizes, base=2, dtype=int)
    print("Learning Rates: {}".format(learning_rates))
    print("Mini-Batch Sizes: {}".format(batch_sizes))

    for filename in DATASETS_PATH:
        x = loadmat(filename)
        Xt, Xv, Ct, Cv, = x['Yt'], x['Yv'], x['Ct'], x['Cv']
        Y = np.argmax(Ct, axis=0)
        datasets = {'train': (Xt, Ct),
                    'test': (Xv, Cv)}

        sns.scatterplot(x=Xt[0], y=Xt[1], hue=Y).set_title("2d visualization of {}".format(filename))

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle("Plots of Accuracy vs. Epoch for Various Learning & Mini-Batch Sizes ".format(filename), fontsize=20)

        for i, (lr, batch_size) in tqdm(enumerate(list(product(learning_rates, batch_sizes))), desc=filename,
                                        total=n_batch_sizes * n_learning_rates):
            input_dimension = Xt.shape[0]
            output_dimension = Ct.shape[0]

            hidden_layers = [] if not with_hidden_layers else HIDDEN_LAYERS_FOR_DATASET[filename]
            net = Network(input_dimension, output_dimension, hidden_layers_sizes=hidden_layers, learning_rate=lr,
                          optimizer=optimizer, gamma=gamma)
            x_plot, y_plots = net.train(Xt, Ct, test_data=datasets, max_epochs=iters, batch_size=batch_size)
            ax = fig.add_subplot(len(learning_rates), len(batch_sizes), i + 1)
            ax.set_title(r"$\alpha={}, bs={}$".format(lr, batch_size), fontsize=16)
            ax.set_ylim([0, 1])
            for ds_name, ds_accuracies in y_plots.items():
                ax.plot(ds_accuracies, label=ds_name)
            ax.legend()
        plt.show()


# run_part_3(with_hidden_layers=False, optimizer=None)
# run_part_3(with_hidden_layers=False, optimizer='momentum', gamma=0.7)
run_part_3(with_hidden_layers=True, optimizer=None)
run_part_3(with_hidden_layers=True, optimizer='momentum', gamma=0.7)
