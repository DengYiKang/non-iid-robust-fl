import torchvision
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt

import sys

from fedlab.utils.dataset import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms

if __name__ == '__main__':
    num_clients = 15
    num_classes = 10
    seed = 50
    alphas = [0.05, 0.1, 1, 10, 100]
    col_names = [f"class{i}" for i in range(num_classes)]
    plt.rcParams['figure.facecolor'] = 'white'
    hist_color = '#4169E1'
    csv_files = ["partition-reports/seed50_clients15_alpha0.05_mnist_noniid_labeldir_clients.csv",
                 "partition-reports/seed50_clients15_alpha0.1_mnist_noniid_labeldir_clients.csv",
                 "partition-reports/seed50_clients15_alpha1_mnist_noniid_labeldir_clients.csv",
                 "partition-reports/seed50_clients15_alpha10_mnist_noniid_labeldir_clients.csv"]
    noniid_labeldir_part_df = [None for i in range(len(csv_files))]
    for i in range(len(csv_files)):
        noniid_labeldir_part_df[i] = pd.read_csv(csv_files[i], header=1)
    # noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
    # noniid_labeldir_part_df = noniid_labeldir_part_df.set_index(None)
    for i in range(len(csv_files)):
        for col in col_names:
            noniid_labeldir_part_df[i][col] = (
                    noniid_labeldir_part_df[i][col] * noniid_labeldir_part_df[i]['Amount']).astype(int)

    # select first 10 clients for bar plot
    fig, ax = plt.subplots(1, len(csv_files), sharex=True, sharey=True, figsize=(20, 3))
    for i in range(len(csv_files)):
        noniid_labeldir_part_df[i][col_names].plot.barh(stacked=True, legend=False, ylabel='Client', ax=ax[i])
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i in range(len(csv_files)):
        plt.subplot(1, len(csv_files), i + 1)
        plt.xlabel('sample num')
        plt.title(f'alpha={alphas[i]}')
    # plt.xlabel('sample num')
    # plt.title("alpha=100")
    # plt.show()
    # plt.savefig(f"./imgs/mnist_noniid_seed{seed}_client{num_clients}_labeldir{alpha}_clients_10.png",
    #             dpi=400, bbox_inches='tight', format='png')
    plt.savefig(f"./imgs/union.png",
                dpi=400, bbox_inches='tight', format='png')
