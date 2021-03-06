#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms


def mnist_one_client_from_designed_frequence(dataset, frequence, size):
    """
    返回一个user的data index，各类的数量由frequence决定
    :param dataset:
    :param frequence:
    :param size:
    :return:
    """
    targets = dataset.targets.numpy()
    idxs = np.array([], dtype='int64')
    for i in range(10):
        if frequence[i] > 0:
            label_idxs = list(np.where(targets == i))[0]
            sample_size = int(size * frequence[i])
            idxs = np.concatenate((idxs, np.random.choice(label_idxs, sample_size, replace=False)), axis=0)
    return idxs


def random_select_on_dict_users(dict_users):
    """
    从dict_users所指定的idxs中随机选出十分之一的sample
    :param dict_users:
    :return:
    """
    dict_sample = {}
    for i in range(len(dict_users)):
        dict_sample[i] = np.random.choice(dict_users[i], int(len(dict_users[i]) / 10), replace=False)
    return dict_sample


def mnist_one_label_select(dataset, label, sample_size):
    """
    从dataset中选取sample_size大小的label，返回idx
    :param dataset:
    :param label:
    :param sample_size
    :return:
    """
    targets = dataset.targets.numpy()
    label_idxs = list(np.where(targets == int(label)))[0]
    sample_size = min(sample_size, len(label_idxs))
    return np.random.choice(label_idxs, sample_size, replace=False)


def mnist_iid(dataset, num_users, sample_size=None):
    """
    Sample I.I.D. client data from MNIST mydataset, each label has sample_size data
    :param dataset:
    :param num_users:
    :param sample_size:
    :return: dict of image index
    """
    if sample_size is None:
        sample_size = int(len(dataset) / num_users / 10)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    targets = dataset.targets.numpy()
    for i in range(num_users):
        for label in range(10):
            idxs = list(np.where(targets == int(label)))[0]
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs, sample_size, replace=False)), axis=0)
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
            rand = random.randint(0, int(num_shards - 1))
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
    targets = dataset.targets.numpy()
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
            rand = random.randint(0, int(num_shards - 1))
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


def cifar_noniid_designed(dataset, cls, per_size):
    """
    生成len(cls)个数据集，第i个数据集拥有cls[i]的类信息，且各个类均衡，每个数据集的数据量大小为train_size。它们之间不一定是iid的。
    :param dataset:
    :param cls: list of list，eg：[[1], [2], [1, 2]]
    :param per_size: 每个数据集的数据量大小
    :return: dict_users，mp, eg:dict_users[i]=[1, 213, 300, ...]
    """
    num_users = len(cls)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = np.array(dataset.targets)
    label_idxs = {}
    for i in range(10):
        label_idxs[i] = np.where(labels == i)[0]
    for i in range(len(cls)):
        num_imgs = int(per_size / len(cls[i]))
        for c in cls[i]:
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(label_idxs[c], num_imgs, replace=False)),
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
