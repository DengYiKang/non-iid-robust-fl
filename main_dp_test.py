import math

import matplotlib, datetime

from utils.record import record_datalist, generate_name_loss, generate_name

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.options import args_parser
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar, AE
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed, mnist_iid_duplicate, \
    random_select_on_dict_users, mnist_one_label_select
from models.Update import LocalUpdate, RebalanceUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img, mnist_test, mnist_all_labels_test
from utils.poisoning import add_attack
from utils.w_glob_cal import geoMed

from fedlab.utils.dataset import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

SAME_VALUE_ATTACK = "same value"
SIGN_FLIPPING_ATTACK = "sign flipping"
GAUSSIAN_NOISY_ATTACK = "gaussian noisy"
NONE_ATTACK = "none attack"

# args=['--seed', '10', '--epochs', '100', '--num_users', '6', '--num_attackers', '2', '--local_bs', '64', '--local_ep', '1', '--data_poisoning', 'all', '--model_poisoning', 'none', '--dir_alpha', '0.1']

if __name__ == "__main__":
    args = args_parser()
    # seed 20 34 50 60 70
    # [7, 9] 3
    source_labels = [7]
    target_label = 1
    if args.source_label != -1:
        source_labels = [args.source_label]
    if args.target_label != -1:
        target_label = args.target_label
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
    # load mydataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        csv_file = f"partition-reports/seed_{args.seed}_clients{args.num_users}_alpha{args.dir_alpha}_tmp.csv"
        part_df = None
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
            # 对异常客户端注入百分之五十的source label，避免无数据可毒的情况
            if args.data_poisoning != "none":
                for idx in range(args.num_attackers):
                    dict_users[idx] = np.append(dict_users[idx], mnist_one_label_select(dataset_train, source_labels[0],
                                                                                        int(len(dict_users[idx]) / 2)))
        # end
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

    partition_report(dataset_train.targets, dict_users,
                     class_num=10,
                     verbose=False, file=csv_file)
    part_df = pd.read_csv(csv_file, header=1)

    # draw distribution
    part_df = part_df.set_index('client')
    col_names = [f"class{i}" for i in range(10)]
    for col in col_names:
        part_df[col] = (part_df[col] * part_df['Amount']).astype(int)
    part_df[col_names].plot.barh(stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.savefig(
        f"imgs/dp的隐蔽性_users{args.num_users}_attackers{args.num_attackers}_seed{args.seed}_alpha{args.dir_alpha}.svg",
        bbox_inches='tight')
    plt.close()
    print("draw end")

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
    data_poisoning_mp_list = []
    # 异常客户端排在前面
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
    # 全局weight，w_overfits[idx]为idx的weight，只保存最近一轮的weight，注意len为num_users
    w_overfits = [w_glob for i in range(args.num_users)]
    net.train()
    w_locals = [w_glob for i in range(args.num_users)]
    loss_test_list = [[] for i in range(args.num_users)]
    acc_test_list = [[] for i in range(args.num_users)]

    # fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        loss_tests = []
        acc_tests = []
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

            net.load_state_dict(w)
            acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, source_labels, target_label)
            acc_test_list[i].append(round(acc_test.item() / 100, 3))
            acc_tests.append(round(acc_test.item() / 100, 3))
            loss_test_list[i].append(round(loss_test, 3))
            loss_tests.append(round(loss_test, 3))
            net.load_state_dict(w_glob)

        w_glob = FedAvg(w_locals)
        net.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_list.append(loss_avg)
        # 每一轮进行测试
        acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, source_labels, target_label)
        acc_list.append(float(acc_test) / 100)
        asr_list.append(asr_test)
        print('Round {:3d}, Average loss {:.3f}, Asr {:.3f}, Acc {:.3f}'.format(iter, loss_avg, asr_test,
                                                                                acc_test.item()))
        print(f"\n\tloss_test: {str(loss_tests)}")
        print(f"\n\tacc_test: {str(acc_tests)}")

    # draw loss_test_list
    plt.figure()
    x = np.linspace(1, args.epochs, args.epochs)
    colors = ['r', 'y', 'c', 'b', 'm', 'g']
    labels = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5']
    for i in range(len(loss_test_list)):
        plt.plot(x, loss_test_list[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Iteration for global update")
    plt.ylabel('Test loss')
    plt.savefig("imgs/dp的隐蔽性_loss.svg")
    plt.close()

    # draw acc_test_list
    plt.figure()
    x = np.linspace(1, args.epochs, args.epochs)
    for i in range(len(acc_test_list)):
        plt.plot(x, acc_test_list[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Iteration for global update")
    plt.ylabel('Test accuracy')
    plt.savefig("imgs/dp的隐蔽性_acc.svg")
    plt.close()

    # draw acc_test
    plt.figure()
    x = np.linspace(1, args.epochs, args.epochs)
    plt.plot(x, acc_list, color=colors[0], linestyle='-', label='global model')
    plt.legend()
    plt.xlabel("Iteration for global update")
    plt.ylabel('Accuracy')
    plt.savefig("imgs/dp的隐蔽性_acc_global.svg")
    plt.close()

    # draw asr_test
    plt.figure()
    x = np.linspace(1, args.epochs, args.epochs)
    yticks = [90 + 2 * i for i in range(6)]
    plt.plot(x, asr_list, color=colors[0], linestyle='-', label='global model')
    plt.legend()
    plt.xlabel("Iteration for global update")
    plt.ylabel('Attack success rate (%)')
    plt.yticks(yticks)
    plt.savefig("imgs/dp的隐蔽性_asr_global.svg")
    plt.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    # plt.savefig(
    #     './result/rebalance_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net.eval()
    # acc_train, loss_train = test_img(net, dataset_train, args)
    # acc_test, loss_test = test_img(net, dataset_test, args)
    acc_train, loss_train, asr_train = mnist_test(net, dataset_train, args, source_labels, target_label)
    acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, source_labels, target_label)
    print("Training accuracy: {:.2f}\nTraining attack success rate: {:.2f}".format(acc_train, asr_train))
    print("\nTesting accuracy: {:.2f}\nTesting attack success rate: {:.2f}".format(acc_test, asr_test))
