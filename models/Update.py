#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, local_ep=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if local_ep is not None:
            self.local_ep = local_ep
        else:
            self.local_ep = self.args.local_ep

    def train(self, net, data_poisoning_mp=None):
        """
        train
        :param net:
        :param data_poisoning_mp:data poisoning，a dict
        :return:
        """
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if data_poisoning_mp is not None:
                    for i in range(len(labels)):
                        labels[i] = data_poisoning_mp[int(labels[i])]
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class RebalanceUpdate(object):
    def __init__(self, args, dataset, labels, idxs):
        self.args = args
        self.dataset = dataset
        self.labels = labels
        self.idxs = idxs
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net):
        """
        one step update
        :param net:
        :return: net.state_dict(), average loss
        """
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for item in self.labels:
            dl = DataLoader(dataset=DatasetSplit(self.dataset, self.idxs[item]), batch_size=self.args.local_bs,
                            shuffle=True)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(dl):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class PersonalizedLoss(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def forward(self, input, target, global_model, per_model):
        loss = F.cross_entropy(input, target) + self.lamda * torch.mean(F.pairwise_distance(per_model, global_model))
        return loss


class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.per_loss_func = PersonalizedLoss(0.05)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, per_net):
        """
        本地训练过程，ditto实现。
        两个优化器，一个用于本地模型，一个用于个性化模型，前者在global update之后刷新为全局模型，后者不会刷新。
        :param net: global model
        :param per_net: personalized model
        :return: net.state_dict(), per_net.state_dict(), loss, per_loss
        """
        kind = 'layer_hidden.weight'
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        per_optimizer = torch.optim.SGD(per_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_per_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_per_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # 本地模型的训练
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # 个性化模型的训练
                per_net.zero_grad()
                per_log_probs = per_net(images)
                per_loss = self.per_loss_func(per_log_probs, labels, net.state_dict()[kind], per_net.state_dict()[kind])
                per_loss.backward()
                per_optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPer Loss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item(), per_loss.item()))
                batch_loss.append(loss.item())
                batch_per_loss.append(per_loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_per_loss.append(sum(batch_per_loss) / len(batch_loss))
        return net.state_dict(), per_net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_per_loss) / len(
            epoch_per_loss)
