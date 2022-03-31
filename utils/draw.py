import numpy as np
import matplotlib.pyplot as plt

from utils.record import read_loss_record
from utils.preprocess import walkFile


def plot(files):
    # save_name = "../result/pic/select/asr/" + files.split("/")[-1][0:-4] + ".svg"
    data = []
    # data = read_loss_record(files)
    for file in files:
        data.append(read_loss_record(file)[0])
    plt.figure()
    x = np.linspace(1, 100, 100)
    colors = ['b', 'g', 'c', 'r', 'm', 'y']
    # RandomValidation Rebalance
    labels = ['FedAvg', 'Krum', 'RFA', 'RFL-MA(ours)']
    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Iterations for global update")
    # Test Accuracy Attack Success Rate
    plt.ylabel("Accuracy")
    # SameValue SignFlipping GaussianNoisy DataPoisoning
    # plt.title("IID MNIST MLP SignFlipping")
    # plt.show()
    result = "acc"
    seed = 20
    alpha = 10.0
    attack_model = "GaussianNoisy"
    plt.savefig(f"../imgs/{result}_seed{seed}_alpha{alpha}_{attack_model}.svg")
    plt.close()


if __name__ == "__main__":
    # file_list = walkFile("../result/asr")
    # for file in file_list:
    #     plot(file)
    # result/acc/seed50_user100_frac0.5_epochs200_dpall_mpnone_modelmlp_datasetmnist_iidFalse.txt
    # fedavg_seed30_user20_attackers6_frac1_epochs100_dpnone_mpsign_flipping_modelmlp_datasetmnist_alpha0.5.txt
    files = []
    files.append(
        "../new_result/acc/fedavg_seed20_user20_attackers6_frac1_epochs100_dpnone_mpgaussian_noisy_modelmlp_datasetmnist_alpha10.0.txt")
    files.append(
        "../new_result/acc/krum_seed20_user20_attackers6_frac1_epochs100_dpnone_mpgaussian_noisy_modelmlp_datasetmnist_alpha10.0.txt")
    files.append(
        "../new_result/acc/rfa_seed20_user20_attackers6_frac1_epochs100_dpnone_mpgaussian_noisy_modelmlp_datasetmnist_alpha10.0.txt")
    files.append(
        "../new_result/acc/ours_seed20_user20_attackers6_frac1_epochs100_dpnone_mpgaussian_noisy_modelmlp_datasetmnist_alpha10.0.txt")
    plot(files)
    # plot("../result/acc/seed50_user100_frac0.1_epochs200_dpall_mpnone_modelmlp_datasetmnist_iidFalse.txt")
