import numpy as np
import matplotlib.pyplot as plt

from utils.record import read_loss_record


def plot(filename):
    data = read_loss_record(filename)
    plt.figure()
    x = np.linspace(1, 200, 200)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    labels = ['FedAvg', 'TrimmedMean', 'GeoMedian', 'Rebalance', 'DataVerification']
    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("asr")
    plt.savefig('test.png')


if __name__ == "__main__":
    plot("../result/asr/seed50_user100_frac0.5_epochs200_dptraining_set_mpnone_modelmlp_datasetmnist_iidTrue.txt")
