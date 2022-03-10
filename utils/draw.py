import numpy as np
import matplotlib.pyplot as plt

from utils.record import read_loss_record
from utils.preprocess import walkFile


def plot(filename):
    save_name = "../result/pic/select/asr/" + filename.split("/")[-1][0:-4] + ".svg"
    data = read_loss_record(filename)
    plt.figure()
    x = np.linspace(1, 200, 200)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    # RandomValidation Rebalance
    labels = ['FedAvg', 'TrimmedMean', 'GeoMedian', 'RandomValidation', 'DataVerification']
    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Communication Rounds")
    # Test Accuracy Attack Success Rate
    plt.ylabel("Attack Success Rate")
    # SameValue SignFlipping GaussianNoisy DataPoisoning
    plt.title("IID MNIST MLP SignFlipping")
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    # file_list = walkFile("../result/asr")
    # for file in file_list:
    #     plot(file)
    # result/acc/seed50_user100_frac0.5_epochs200_dpall_mpnone_modelmlp_datasetmnist_iidFalse.txt
    plot("../result/asr/seed50_user100_frac0.5_epochs200_dpnone_mpsign_flipping_modelmlp_datasetmnist_iidTrue.txt")
