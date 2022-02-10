import math

import matplotlib, datetime

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.options import args_parser
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar, AE
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed, mnist_iid_duplicate
from models.Update import LocalUpdate, RebalanceUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img, mnist_test, mnist_all_labels_test
from utils.poisoning import add_attack
from utils.w_glob_cal import geoMed

SAME_VALUE_ATTACK = "same value"
SIGN_FLIPPING_ATTACK = "sign flipping"
GAUSSIAN_NOISY_ATTACK = "gaussian noisy"
NONE_ATTACK = "none attack"

if __name__ == "__main__":
    args = args_parser()
    # seed 20 34 50 60 70
    source_labels = [7, 9]
    target_label = 3
    # byzantine比例
    byzantine_proportion = 0.3
    # drop out proportion
    drop_out_proportion = 0.2
    # 根据错误率排序，错误率前多少的需要进行矫正
    alpha = 3
    # 每轮随机挑选的client数量
    m = max(int(args.frac * args.num_users), 1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    # args, dataset
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 定义所有user的类信息
    cls = []
    if args.iid:
        # iid的cls
        for i in range(args.num_users):
            tmp = []
            for j in range(0, 10):
                tmp.append(j)
            cls.append(tmp)
    else:
        # 随机的cls
        for i in range(args.num_users):
            tmp = set()
            for rand in range(random.randint(1, 10)):
                tmp.add(random.randint(0, 9))
            cls.append(list(tmp))
    # load mydataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if cls is not None:
            dict_users = mnist_noniid_designed(dataset_train, cls, 1000)
        elif args.iid:
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
    # 共享的测试集，用于data validation，测试集的数据取自client的本地数据
    # 这里简化为：测试集的数据分布与对应client的本地数据的相同；
    # 验证集是从训练集中抽样得到的
    # todo:这里的数据分布不是完全的non-iid的，每个类的数据量是相等的
    shared_dataset_test_idx = mnist_noniid_designed(dataset_train, cls, 100)
    # 用于测试各个类的准确度
    balanced_test_set_idx = mnist_iid_duplicate(dataset_train, 1, [i for i in range(10)], 1000)[0]
    # rebalance的训练集，每个类的数据量为100，rebalance_train_set_idx[i]用于对第i标签高错误率的补偿训练
    rebalance_train_set_idx = mnist_noniid_designed(dataset_train, [[i] for i in range(10)], 100)

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net)
    # copy weights
    w_glob = net.state_dict()
    # 1、各种参数
    # list of data poisoning map
    attack_mp = {}
    for i in range(0, 10):
        if i in source_labels:
            attack_mp[i] = target_label
        else:
            attack_mp[i] = i
        # attack_mp[i] = random.randint(0, 5)
        # attack_mp[i] = random.randint(0, 9)
        # attack_mp[i] = 1
    data_poisoning_mp_list = []
    for idx in range(args.num_users):
        if idx < args.num_users * byzantine_proportion:
            data_poisoning_mp_list.append(attack_mp)
        else:
            data_poisoning_mp_list.append(None)
    # 初始化记录
    # loss_per_client[idx]为list，记录不同轮数时某个user的loss值
    loss_per_client = {}
    # 记录全局模型的loss的变化过程，用于绘图
    loss_train_list = []
    for t in range(args.num_users):
        loss_per_client[t] = []
    print("Aggregation over all clients")
    # 全局weight，w_overfits[idx]为idx的weight，只保存最近一轮的weight，注意len为num_users
    w_overfits = [w_glob for i in range(args.num_users)]
    net.train()
    # 所有client进行本地训练直到过拟合，client上传模型参数到中央服务器
    for idx in range(args.num_users):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=shared_dataset_test_idx[idx], local_ep=30)
        w, loss = local.train(net=copy.deepcopy(net),
                              data_poisoning_mp={True: data_poisoning_mp_list[idx], False: None}[
                                  args.data_poisoning == "all"])
        w_overfits[idx] = copy.deepcopy(w)
    # 全局weight，w_locals[idx]为最近一轮idx序号的weight，只保存最近一轮的weight，这时的len为m
    w_locals = [w_glob for i in range(m)]
    rebalance_labels = {}
    # fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        # data validation score list
        f_scores = []
        byzantine_users = np.random.choice(range(int(args.num_users * byzantine_proportion)),
                                           math.ceil(m * byzantine_proportion), replace=False)
        normal_users = np.random.choice(range(int(args.num_users * byzantine_proportion), args.num_users),
                                        m - len(byzantine_users), replace=False)
        idxs_users = np.concatenate((byzantine_users, normal_users))
        idxs_users = np.sort(idxs_users)
        for i, idx in enumerate(idxs_users):
            # 训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net).to(args.device),
                                  data_poisoning_mp={True: None, False: data_poisoning_mp_list[idx]}[
                                      args.data_poisoning == "none"])
            loss_per_client[idx].append(loss)
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            if idx < args.num_users * byzantine_proportion:
                # 序号前指定的比例为byzantine，将上传的w替换。model poisoning
                if args.model_poisoning == "same_value":
                    w_locals[i] = add_attack(w_locals[i], SAME_VALUE_ATTACK)
                elif args.model_poisoning == "sign_flipping":
                    w_locals[i] = add_attack(w_locals[i], SIGN_FLIPPING_ATTACK)
                elif args.model_poisoning == "gaussian_noisy":
                    w_locals[i] = add_attack(w_locals[i], GAUSSIAN_NOISY_ATTACK)
                else:
                    pass
        # data verification, f_scores
        for i, idx in enumerate(idxs_users):
            accuracy, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w_locals[i],
                                            dataset=dataset_train, args=args,
                                            idx=shared_dataset_test_idx[idx],
                                            data_poisoning_mp=
                                            {True: data_poisoning_mp_list[idx], False: None}[
                                                args.data_poisoning == "all"])
            f_scores.append(test_loss)
        drop_out_idxs = np.argsort(f_scores)[::-1][:int(drop_out_proportion * m)]
        for i in drop_out_idxs:
            f_scores[i] = 0
        # 除去异常的client之外，各个client的贡献均衡
        for i in range(len(f_scores)):
            if f_scores[i] > 0:
                f_scores[i] = 1
        f_sum = np.sum(f_scores)
        f_scores = [item / f_sum for item in f_scores]
        # rebalance
        for i, idx in enumerate(idxs_users):
            if f_scores[i] == 0:
                continue
            net.load_state_dict(w_locals[i])
            if idx not in rebalance_labels:
                # 计算需要补偿的类别
                asr = mnist_all_labels_test(net_g=copy.deepcopy(net).to(args.device),
                                            dataset=dataset_train, idxs=balanced_test_set_idx, args=args)
                rebalance_labels[idx] = np.argsort(asr)[::-1][:alpha]
            # 补偿训练
            re_update = RebalanceUpdate(args=args, dataset=dataset_train, labels=rebalance_labels[idx],
                                        idxs=rebalance_train_set_idx)
            w, loss = re_update.train(net=copy.deepcopy(net).to(args.device))
            w_locals[i] = w
        # rebalance end
        w_locals_normal = []
        for i in range(m):
            if f_scores[i] != 0:
                w_locals_normal.append(w_locals[i])
        w_glob = geoMed(w_locals=w_locals_normal, args=args, kind=None, groups=10)
        # FedAvg，这一行作为对照,取消注释后就变为FedAvg
        # w_glob = FedAvg(w_locals)
        net.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    # plt.savefig('./model/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net.eval()
    # acc_train, loss_train = test_img(net, dataset_train, args)
    # acc_test, loss_test = test_img(net, dataset_test, args)
    acc_train, loss_train, asr_train = mnist_test(net, dataset_train, args, source_labels, target_label)
    acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, source_labels, target_label)
    print("Training accuracy: {:.2f}\nTraining attack success rate: {:.2f}".format(acc_train, asr_train))
    print("\nTesting accuracy: {:.2f}\nTesting attack success rate: {:.2f}".format(acc_test, asr_test))
