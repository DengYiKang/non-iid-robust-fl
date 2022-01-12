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
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img, mnist_test
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
    byzantine_proportion = 0.2
    # top k sims
    top_proportion = 0.1
    # drop out proportion
    drop_out_proportion = 0.2
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
    # 共享的测试集，用于data validation，测试集的数据取自client的本地数据，
    # 这里简化为：测试集的数据分布与对应client的本地数据的相同；
    # todo:在最后的测试中，需要把共享的数据剔除
    shared_dataset_test_idx = mnist_noniid_designed(dataset_test, cls, 100)

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
    # ae的标准化参数
    ae_mean = -0.0004
    ae_std = 0.1251
    # 过拟合的特征
    features = []
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
    # 全局更新计算式子中的新旧权值
    alpha = 0
    # 两种分数的加权
    beta = 0.5
    # 初始化记录
    # loss_per_client[idx]为list，记录不同轮数时某个user的loss值
    loss_per_client = {}
    # 记录全局模型的loss的变化过程，用于绘图
    loss_train_list = []
    kind = 'layer_hidden.weight'
    for t in range(args.num_users):
        loss_per_client[t] = []
    print("Aggregation over all clients")
    # 全局weight，w_locals[idx]为idx的weight，只保存最近一轮的weight
    w_locals = [w_glob for i in range(args.num_users)]
    net.train()
    # 加载anomaly detection model以及优化器参数，基于AE，这个AE也被用作过拟合模型的特征提取
    ae_model_path = 'anomaly_detection/model/mnist_mlp_dimIn500_size19360_batch5_seed10_loss0.055.pth'
    ae_net = AE().to(args.device)
    optimizer = torch.optim.SGD(ae_net.parameters(), lr=0.001)
    checkpoint = torch.load(ae_model_path)
    ae_net.load_state_dict(checkpoint['net'])
    # 所有client进行本地训练直到过拟合，client上传模型参数到中央服务器，中央服务器进行特征提取
    for idx in range(args.num_users):
        local = LocalUpdate(args=args, dataset=dataset_test, idxs=shared_dataset_test_idx[idx], local_ep=30)
        w, loss = local.train(net=copy.deepcopy(net),
                              data_poisoning_mp={True: data_poisoning_mp_list[idx], False: None}[
                                  args.data_poisoning == "all"])
        w_locals[idx] = copy.deepcopy(w)
    ae_net.eval()
    for idx in range(args.num_users):
        origin = (w_locals[idx][kind].view(1, -1) - ae_mean) / ae_std
        encoded, decoded = ae_net(origin)
        features.append(encoded)
    # 计算client之间的相似度
    # sims[x][y]越小，x与y越相似
    sims = [[None for item in range(args.num_users)] for _ in range(args.num_users)]
    for x in range(args.num_users):
        for y in range(args.num_users):
            if x == y:
                sims[x][y] = 0
            else:
                dist = F.pairwise_distance(features[x], features[y], p=2)
                sims[x][y] = dist.item()
    # 正则化
    sims_d1 = [val for item in sims for val in item]
    sims_mean = np.mean(sims_d1)
    sims_std = np.std(sims_d1)
    sims = [[(sims[i][j] - sims_mean) / sims_std for j in range(len(sims))] for i in range(len(sims))]
    # 为每个client生成有序的列表, sims_in_order[x]为保存user序号的有序列表，sims_in_order[x][0]为与x最相似的user序号
    print("sims:")
    for i in range(len(sims)):
        print(sims[i])
    sims_in_order = []
    for idx in range(args.num_users):
        sims_in_order.append(np.argsort(sims[idx]).tolist())
    top_sims = [np.sum(item[:int(top_proportion * args.num_users)]) for item in sims]
    drop_out_idxs_top_sims = np.argsort(top_sims)[:int(top_proportion * args.num_users)]
    # fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        # data validation score list
        f_scores = []
        for idx in range(args.num_users):
            # 训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net).to(args.device),
                                  data_poisoning_mp={True: None, False: data_poisoning_mp_list[idx]}[
                                      args.data_poisoning == "none"])
            loss_per_client[idx].append(loss)
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            if idx < args.num_users * byzantine_proportion:
                # 序号前指定的比例为byzantine，将上传的w替换。model poisoning
                w_locals[idx] = add_attack(w_locals[idx], GAUSSIAN_NOISY_ATTACK)
                pass
        # data verification, f_scores
        for idx in range(args.num_users):
            accuracy, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w_locals[idx],
                                            dataset=dataset_test, args=args,
                                            idx=shared_dataset_test_idx[idx],
                                            data_poisoning_mp=
                                            {True: data_poisoning_mp_list[idx], False: None}[
                                                args.data_poisoning == "all"])
            f_scores.append(test_loss)
        f_mean = np.mean(f_scores)
        f_std = np.std(f_scores)
        f_scores = [1 / ((math.exp((item - f_mean) / f_std)) ** 2) for item in f_scores]
        # 评分规则下等比例地移除低分
        drop_out_idxs = np.union1d(drop_out_idxs_top_sims,
                                   np.argsort(f_scores)[:int(drop_out_proportion * args.num_users)])
        # drop_out_idxs = np.argsort(f_scores)[:int(drop_out_proportion * args.num_users * 2)]
        for idx in drop_out_idxs:
            f_scores[idx] = 0
        f_sum = np.sum(f_scores)
        # 正则化
        for idx in range(args.num_users):
            f_scores[idx] /= f_sum
        # 除去异常的client之外，各个client的贡献均衡
        for idx in range(len(f_scores)):
            if f_scores[idx] > 0:
                f_scores[idx] = 1
        f_sum = np.sum(f_scores)
        f_scores = [item / f_sum for item in f_scores]
        # 2.6、更新全局model
        # for k in w_glob.keys():
        #     w_glob[k] = alpha * w_glob[k]
        # for i in range(args.num_users):
        #     for k in w_glob.keys():
        #         w_glob[k] += (1 - alpha) * f_scores[i] * w_locals[i][k]
        w_locals_normal = []
        for idx in range(args.num_users):
            if f_scores[idx] != 0:
                w_locals_normal.append(w_locals[idx])
        w_glob = geoMed(w_locals=w_locals_normal, args=args, kind=kind, groups=10)
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
