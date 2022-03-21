#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from models.Update import DatasetSplit


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def mnist_all_labels_test(net_g, dataset, idxs, args):
    """
    测试，返回各个标签的预测错误率，a标签的错误率=真实标签为a但被错误预测的数量/真实标签为a的数量
    :param net_g:
    :param dataset:
    :param args:
    :return:asr, 一个list，长度为10,表示各个标签的预测错误率
    """
    asr = [None for i in range(10)]
    net_g.eval()
    # testing
    if idxs is None:
        data_loader = DataLoader(dataset, batch_size=args.bs)
    else:
        data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs)
    cnt_source = [0 for i in range(10)]
    cnt_mislabel = [0 for i in range(10)]
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        for i in range(len(target)):
            cnt_source[int(target[i])] += 1
        log_probs = net_g(data)
        prob_labels = [np.argsort(item.tolist())[-1] for item in log_probs]
        for i in range(len(target)):
            if prob_labels[i] != target[i]:
                cnt_mislabel[int(target[i])] += 1
    for i in range(10):
        # 因为数据集是randomly sample的，可能存在没有选到这一类标签的情况
        if cnt_source[i] == 0:
            asr[i] = 100.00
            # print("\n[bug] balanced test set has no {}-label\n".format(i))
        else:
            asr[i] = 100.00 * cnt_mislabel[i] / cnt_source[i]
    return asr


def mnist_test(net_g, dataset, args, source_labels, target_label):
    """
    测试，返回acc和asr, asr=真实标签为source_labels但被错误预测为target_label的标签数量/真实标签为source_labels的标签数量
    :param net_g:
    :param dataset:
    :param args:
    :param source_labels: asr中的source labels，为list
    :param target_label: asr中的target label
    :return:acc，test loss, asr
    """
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)
    cnt_source = 0
    cnt_mislabel = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        for i in range(len(target)):
            if int(target[i]) in source_labels:
                cnt_source += 1
        log_probs = net_g(data)
        prob_labels = [np.argsort(item.tolist())[-1] for item in log_probs]
        for i in range(len(target)):
            if int(target[i]) in source_labels and prob_labels[i] == target_label:
                cnt_mislabel += 1
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print(f"cnt_mislabel: {cnt_mislabel}, cnt_source: {cnt_source}, acc: {accuracy}")
    asr = 100.00 * cnt_mislabel / cnt_source
    if args.verbose:
        print(
            '\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\nAttack Success Rate: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(data_loader.dataset), accuracy, cnt_mislabel, cnt_source, asr))
    return accuracy, test_loss, asr


def brca_test(net, w, dataset, args, idx, data_poisoning_mp=None):
    """
    brca中的共享测试
    :param net:
    :param w:
    :param dataset:
    :param args:
    :param idx:
    :param data_poisoning_mp:
    :return: acc, test_loss
    """
    net.load_state_dict(w)
    net.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset, idx), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        # 共享测试集上的data poisoning
        if data_poisoning_mp is not None:
            for i in range(len(target)):
                target[i] = data_poisoning_mp[int(target[i])]
        log_probs = net(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
