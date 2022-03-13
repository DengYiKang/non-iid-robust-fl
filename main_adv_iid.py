import math

import matplotlib, datetime
from fedlab.utils.dataset import MNISTPartitioner

from utils.record import record_datalist, generate_name
from utils.w_glob_cal import geoMed

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
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed, mnist_one_label_select, \
    random_select_on_dict_users
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img, mnist_test
from utils.poisoning import add_attack

SAME_VALUE_ATTACK = "same value"
SIGN_FLIPPING_ATTACK = "sign flipping"
GAUSSIAN_NOISY_ATTACK = "gaussian noisy"
NONE_ATTACK = "none attack"

if __name__ == "__main__":
    args = args_parser()
    # [7, 9] 3
    source_labels = [7]
    target_label = 1
    # drop out proportion
    drop_out_proportion = 0.2
    # data verification阶段，相似度前gamma比例的不被选取
    gamma = 0.2
    # data verification阶段，随机选取的范围长度
    gamma_len = 0.2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    # args, dataset
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load mydataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            iid_part = MNISTPartitioner(dataset_train.targets,
                                        num_clients=args.num_users,
                                        partition="iid",
                                        seed=args.seed)
            dict_users = iid_part.client_dict
        else:
            noniid_labeldir_part = MNISTPartitioner(dataset_train.targets, args.num_users, partition="noniid-labeldir",
                                                    dir_alpha=args.dir_alpha,
                                                    seed=args.seed)
            dict_users = noniid_labeldir_part.client_dict
            # 对异常客户端注入百分之二十的source label，避免无数据可毒的情况
            if args.data_poisoning != "none":
                for idx in range(args.num_attackers):
                    dict_users[idx] = np.append(dict_users[idx], mnist_one_label_select(dataset_train, source_labels[0],
                                                                                        int(len(dict_users[idx]) / 5)))
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
    # 共享的测试集，用于data validation，测试集的数据取自client的本地数据，
    shared_dataset_test_idx = mnist_iid(dataset_train, args.num_users, 50)

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
    # list of data poisoning map
    attack_mp = {}
    for i in range(0, 10):
        if i in source_labels:
            attack_mp[i] = target_label
        else:
            attack_mp[i] = i
    data_poisoning_mp_list = []
    for idx in range(args.num_users):
        if idx < args.num_attackers:
            data_poisoning_mp_list.append(attack_mp)
        else:
            data_poisoning_mp_list.append(None)
    # 初始化记录
    # loss_per_client[idx]为list，记录不同轮数时某个user的loss值
    loss_per_client = {}
    # 记录全局模型的loss的变化过程，用于绘图
    loss_train_list = []
    acc_list = []
    asr_list = []
    kind = 'layer_hidden.weight'
    for t in range(args.num_users):
        loss_per_client[t] = []
    print("Aggregation over all clients")
    # 全局weight，w_locals[idx]为idx的weight，只保存最近一轮的weight
    w_overfits = [w_glob for i in range(args.num_users)]
    net.train()
    # 所有client进行本地训练直到过拟合，client上传模型参数到中央服务器，中央服务器进行特征提取
    for idx in range(args.num_users):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=shared_dataset_test_idx[idx], local_ep=30)
        w, loss = local.train(net=copy.deepcopy(net),
                              data_poisoning_mp={True: data_poisoning_mp_list[idx], False: None}[
                                  args.data_poisoning == "all"])
        w_overfits[idx] = copy.deepcopy(w)
    # 计算w_overfits之间的cos相似度
    sims = np.zeros((args.num_users, args.num_users))
    for x in range(args.num_users):
        for y in range(0, x + 1):
            if x == y:
                sims[x][y] = 1.0
            else:
                cosim = torch.cosine_similarity(w_overfits[x][kind].view(1, -1), w_overfits[y][kind].view(1, -1))
                sims[x][y] = sims[y][x] = cosim.item()
    # 为每个client生成有序的列表, sims_in_order[x]为保存user序号的有序列表，sims_in_order[x][0]为与x最相似的user序号
    sims_in_order = []
    for idx in range(args.num_users):
        sims_in_order.append(np.argsort(sims[idx])[::-1].tolist())
    # 全局weight，w_locals[idx]为最近一轮idx序号的weight，只保存最近一轮的weight，这时的len为m
    w_locals = [w_glob for i in range(args.num_users)]
    # fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        # data validation score list
        f_scores = []
        for i in range(args.num_users):
            # 训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[i])
            w, loss = local.train(net=copy.deepcopy(net).to(args.device),
                                  data_poisoning_mp={True: None, False: data_poisoning_mp_list[i]}[
                                      args.data_poisoning == "none"])
            loss_per_client[i].append(loss)
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            if i < args.num_attackers:
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
        # 对每个client，随机选取与它相似的client共享的数据集做验证集
        for i in range(args.num_users):
            random_idx = random.randint(int(gamma * args.num_users),
                                        int((gamma + gamma_len) * args.num_users))
            verification_set_idx = sims_in_order[i][random_idx]
            accuracy, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w_locals[i],
                                            dataset=dataset_train, args=args,
                                            idx=shared_dataset_test_idx[verification_set_idx],
                                            data_poisoning_mp=
                                            {True: data_poisoning_mp_list[verification_set_idx], False: None}[
                                                args.data_poisoning == "all"])
            f_scores.append(test_loss)
        # 只根据data verification取出异常客户端
        drop_out_idxs = np.argsort(f_scores)[::-1][:int(drop_out_proportion * args.num_users)]
        f_scores = np.array(f_scores)
        f_scores[drop_out_idxs] = 0
        # 除去异常的client之外，各个client的贡献均衡
        f_scores[f_scores != 0] = 1
        f_scores = f_scores / np.sum(f_scores)
        w_locals_normal = []
        for i in range(args.num_users):
            if f_scores[i] >= 0:
                w_locals_normal.append(w_locals[i])
        w_glob = geoMed(w_locals=w_locals_normal, args=args, kind=None, groups=int(len(w_locals_normal) / 5))
        # FedAvg，这一行作为对照,取消注释后就变为FedAvg，attack生效
        # w_glob = FedAvg(w_locals)
        net.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
        # 每一轮进行测试
        net.eval()
        acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, source_labels, target_label)
        acc_list.append(float(acc_test))
        asr_list.append(asr_test)
        net.train()

    acc_list = [round(float(item) / 100, 3) for item in acc_list]

    asr_list = [round(float(item) / 100, 5) for item in asr_list]
    # # save loss list
    # record_datalist(loss_train_list,
    #                 generate_name(args.seed, args.num_users, args.frac, args.epochs, args.data_poisoning,
    #                               args.model_poisoning, args.iid, args.model, args.dataset, "loss"))
    # # save acc list
    # record_datalist(acc_list, generate_name(args.seed, args.num_users, args.frac, args.epochs, args.data_poisoning,
    #                                         args.model_poisoning, args.iid, args.model, args.dataset, "acc"))
    # # save asr list
    # record_datalist(asr_list, generate_name(args.seed, args.num_users, args.frac, args.epochs, args.data_poisoning,
    #                                         args.model_poisoning, args.iid, args.model, args.dataset, "asr"))
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
