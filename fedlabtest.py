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
    num_clients = 5
    num_classes = 10
    seed = 20
    alpha = 0.1
    col_names = [f"class{i}" for i in range(num_classes)]
    plt.rcParams['figure.facecolor'] = 'white'
    hist_color = '#4169E1'
    trainset = torchvision.datasets.MNIST(root="./data/mnist/", train=True, download=True)
    # noniid_labeldir_part.client_dict替换dict_users
    noniid_labeldir_part = MNISTPartitioner(trainset.targets,
                                            num_clients=num_clients,
                                            partition="noniid-labeldir",
                                            dir_alpha=alpha,
                                            seed=seed)

    csv_file = "./partition-reports/" + "seed" + str(seed) + "_clients" + str(
        num_clients) + "_alpha" + str(
        alpha) + "_mnist_noniid_labeldir_clients.csv"
    partition_report(trainset.targets, noniid_labeldir_part.client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)
    noniid_labeldir_part_df = pd.read_csv(csv_file, header=1)
    noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
    for col in col_names:
        noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(int)

    # select first 10 clients for bar plot
    noniid_labeldir_part_df[col_names].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.savefig(f"./imgs/noniid对验证集loss的影响_users{num_clients}_seed{seed}_alpha{alpha}.svg",
                bbox_inches='tight')
