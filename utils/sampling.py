#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST mydataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST mydataset randomly
    :param dataset:
    :param num_users:
    :return:
    """
    # 总数据量60k，每个类的数量约为6k
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs为按照label排好序后的索引
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # 每个user拥有2*num_imgs=600张图片，最多拥有两类或者一类
        # replace=False表示不可以取相同数字
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_iid_duplicate(dataset, num_users, class_list, train_size):
    """
    生成num_users个数据集，它们持有class_list中所定义的类数据，且各类数据均衡，它们之间是iid的。
    :param dataset:
    :param num_users:
    :param class_list:[1, 2]表示拥有1、2两个类的数据，元素取值范围：0~9
    :param train_size:训练集的大小
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs为按照label排好序后的索引
    idxs = idxs_labels[0, :]
    num_per_class = int(len(dataset) / 10)

    for i in range(num_users):
        num_imgs = int(train_size / len(class_list))
        num_shards = num_per_class / num_imgs
        for cls in class_list:
            offset = cls * num_per_class
            rand = random.randint(0, num_shards - 1)
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[offset + rand * num_imgs:offset + (rand + 1) * num_imgs]),
                axis=0)
    return dict_users


def mnist_noniid_designed(dataset, cls, per_size):
    """
    生成len(cls)个数据集，第i个数据集拥有cls[i]的类信息，且各个类均衡，每个数据集的数据量大小为train_size。它们之间不一定是iid的。
    :param dataset:
    :param cls:list of list，eg：[[1], [2], [1, 2]]
    :param per_size:每个数据集的数据量大小
    :return:dict_users，mp, eg:dict_users[i]=[1, 213, 300, ...]
    """
    num_users = len(cls)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs为按照label排好序后的索引
    idxs = idxs_labels[0, :]
    num_per_class = int(len(dataset) / 10)

    for i in range(len(cls)):
        # 每个类有num_imgs个数据量
        num_imgs = int(per_size / len(cls[i]))
        num_shards = num_per_class / num_imgs
        for c in cls[i]:
            offset = c * num_per_class
            rand = random.randint(0, num_shards - 1)
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[offset + rand * num_imgs:offset + (rand + 1) * num_imgs]),
                axis=0)
    return dict_users


def mnist_noniid_only_one_class(dataset, num_users, class_idx):
    """
    Sample client data from MNIST dataset, each user has the same one class
    :param dataset:
    :param num_users:
    :param class_idx:0~9
    :return:
    """
    num_shards, num_imgs = 10, 600
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs为按照label排好序后的索引
    idxs = idxs_labels[0, :]
    num_per_class = len(dataset) / 10

    # divide and assign
    for i in range(num_users):
        # 每个user拥有1*num_imgs=600张图片，只能拥有class_idx这一个类的图片
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        offset = class_idx * num_per_class
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[offset + rand * num_imgs:offset + (rand + 1) * num_imgs]),
                axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 mydataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
