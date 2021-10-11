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


def union():
    files = walkFile("./save/data/encoder/input")
    data = []
    for f in files:
        data.extend(torch.load(f))
    torch.save(data, './save/data/encoder/tot/size_40k.pt')


def split():
    data = torch.load('./save/data/encoder/tot/size_40k.pt')
    offset = int(0.8 * len(data))
    random.shuffle(data)
    train = data[:offset]
    test = data[offset:]
    torch.save(train, './save/data/encoder/train/size_32k.pt')
    torch.save(test, './save/data/encoder/test/size_8k.pt')


def cal_standard():
    data = torch.load('./save/data/encoder/tot/size_40k.pt')
    t = data[0]
    t.cuda()
    for idx, item in enumerate(data):
        if idx != 0:
            t = torch.cat([t.cuda(), item.cuda()])
    t = t.view(1, -1)
    mean = t.mean()
    std = t.std()
    return mean, std


def standard(mean, std):
    train_data = torch.load('./save/data/encoder/train/size_32k.pt')
    test_data = torch.load('./save/data/encoder/test/size_8k.pt')
    for idx in range(len(train_data)):
        train_data[idx] = (train_data[idx].cuda() - mean) / std
    for idx in range(len(test_data)):
        test_data[idx] = (test_data[idx].cuda() - mean) / std
    torch.save(train_data, './save/data/encoder/train/size_32k_standard.pt')
    torch.save(test_data, './save/data/encoder/test/size_8k_standard.pt')


if __name__ == '__main__':
    union()
    split()
    mean, std = cal_standard()
    standard(mean.cuda(), std.cuda())
