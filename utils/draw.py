import numpy as np
import matplotlib.pyplot as plt

from utils.record import read_loss_record
from utils.preprocess import walkFile


def plot(filename):
    save_name = "../result/pic/select/loss/" + filename.split("/")[-1][0:-4] + ".svg"
    data = read_loss_record(filename)
    plt.figure()
    x = np.linspace(1, 200, 200)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    labels = ['FedAvg', 'TrimmedMean', 'GeoMedian', 'Rebalance', 'DataVerification']
    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    # plt.ylabel("Test Accuracy")
    plt.title("Non-IID MNIST MLP GaussianNoisy")
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    # file_list = walkFile("../result/asr")
    # for file in file_list:
    #     plot(file)
    plot("../result/loss/seed50_user100_frac0.5_epochs200_dpnone_mpgaussian_noisy_modelmlp_datasetmnist_iidFalse.txt")
