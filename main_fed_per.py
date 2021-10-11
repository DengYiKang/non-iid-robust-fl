#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy  # 用于联邦学习全局模型的复制过程
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load mydataset and split users
    if args.dataset == 'mnist':
        # 0.1307是MNIST数据集的全局平均值，0.3081是MNIST的标准偏差
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized mydataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    loss_per_client = {}
    kind = 'layer_hidden.weight'

    for t in range(args.num_users):
        loss_per_client[t] = []
    print("Aggregation over all clients")
    w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        m = max(args.num_users, 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = range(args.num_users)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            loss_per_client[idx].append(loss)
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot global loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.svg'.format(args.mydataset, args.model, args.epochs, args.frac, args.iid))

    # plot local loss curve
    # for client_id, loss_list in loss_per_client.items():
    #     plt.figure()
    #     plt.plot(range(len(loss_list)), loss_list)
    #     plt.ylabel('train_loss')
    #     if args.iid:
    #         plt.savefig(
    #             './save/local/iid/client_{}_{}_{}_{}_C{}_iid{}.svg'.format(client_id, args.mydataset, args.model,
    #                                                                        args.epochs,
    #                                                                        args.frac, args.iid))
    #     else:
    #         plt.savefig(
    #             './save/local/non-iid/client_{}_{}_{}_{}_C{}_iid{}.svg'.format(client_id, args.mydataset, args.model,
    #                                                                            args.epochs,
    #                                                                            args.frac, args.iid))
    #     plt.close()

    # plot local weight
    # x = []
    # y = []
    # x_major_locator = MultipleLocator(0.005)
    # y_major_locator = MultipleLocator(0.01)
    # for i in range(10):
    #     x.append([])
    #     y.append([])
    #     for idx in range(args.num_users):
    #         x[i].append(w_locals[idx][kind][i].mean().item())
    #         y[i].append(w_locals[idx][kind][i].std().item())
    #     plt.figure()
    #     plt.title('hidden_layer statistics of #' + str(i))
    #     plt.xlabel('mean')
    #     plt.ylabel('std')
    #     ax = plt.gca()
    #     ax.xaxis.set_major_locator(x_major_locator)
    #     ax.yaxis.set_major_locator(y_major_locator)
    #     plt.xlim(-0.010, 0.030)
    #     plt.ylim(0.04, 0.10)
    #     plt.scatter(x[i], y[i])
    #     plt.savefig(
    #         './save/local/weight/non-iid/kind_{}_{}_{}_{}_iid{}.svg'.format(i, args.mydataset, args.model,
    #                                                                         args.local_ep, args.iid))
    #     plt.close()
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
