#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random

import torch


def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        file_list = []
        # 遍历文件
        for f in files:
            file_list.append(os.path.join(root, f))
        return file_list


def union(input_path, tot_path):
    files = walkFile(input_path)
    data = []
    for f in files:
        data.extend(torch.load(f))
    torch.save(data, tot_path)


def split(tot_path, train_path, test_path):
    data = torch.load(tot_path)
    offset = int(0.8 * len(data))
    random.shuffle(data)
    train = data[:offset]
    test = data[offset:]
    torch.save(train, train_path)
    torch.save(test, test_path)


def cal_standard(tot_path):
    data = torch.load(tot_path)
    t = data[0]
    t.cuda()
    for idx, item in enumerate(data):
        if idx != 0:
            t = torch.cat([t.cuda(), item.cuda()])
    t = t.view(1, -1)
    mean = t.mean()
    std = t.std()
    return mean, std


def standard(mean, std, train_path, test_path, train_standard_path, test_standard_path):
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    for idx in range(len(train_data)):
        train_data[idx] = (train_data[idx].cuda() - mean) / std
    for idx in range(len(test_data)):
        test_data[idx] = (test_data[idx].cuda() - mean) / std
    torch.save(train_data, train_standard_path)
    torch.save(test_data, test_standard_path)


def preprocess(input_path, tot_path, train_path, test_path, train_standard_path, test_standard_path):
    union(input_path, tot_path)
    split(tot_path, train_path, test_path)
    mean, std = cal_standard(tot_path)
    standard(mean, std, train_path, test_path, train_standard_path, test_standard_path)
