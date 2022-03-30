import math

import matplotlib, datetime

from utils.record import record_datalist, generate_name_loss, generate_name, read_loss_record

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
    random_select_on_dict_users, mnist_one_label_select, mnist_one_client_from_designed_frequence
from models.Update import LocalUpdate, RebalanceUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img, mnist_test, mnist_all_labels_test
from utils.poisoning import add_attack
from utils.w_glob_cal import geoMed

from fedlab.utils.dataset import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

# noniid对验证集loss的影响

def draw(filename):
    data = read_loss_record(filename)
    plt.figure()
    x = np.linspace(1, 100, 100)
    colors = ['r', 'g', 'c', 'm', 'y']
    # RandomValidation Rebalance
    labels = ['client0', 'client1', 'client2', 'client3', 'client4']
    for i in range(len(data)):
        plt.plot(x, data[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Iterations for global update")
    # Test Accuracy Attack Success Rate
    plt.ylabel("Validation Loss")
    # SameValue SignFlipping GaussianNoisy DataPoisoning
    # plt.title("IID MNIST MLP SignFlipping")
    # plt.show()
    seed = 20
    alpha = 0.1
    iid = False
    plt.savefig(f"seed{seed}_alpha{alpha}_iid{iid}_noniid对验证集loss的影响.svg")
    plt.close()


if __name__ == "__main__":
    draw("3/noniid对验证集loss的影响/seed20_alpha0.1_iidFalse_users5.txt")
    args = args_parser()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    # args, dataset
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load mydataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    csv_file = f"../partition-reports/seed_{args.seed}_clients10_alpha{args.dir_alpha}_tmp.csv"
    part_df = None
    # sample users
    if args.iid:
        iid_part = MNISTPartitioner(dataset_train.targets,
                                    num_clients=args.num_users,
                                    partition="iid",
                                    seed=args.seed)
        dict_users = iid_part.client_dict
        partition_report(dataset_train.targets, dict_users,
                         class_num=10,
                         verbose=False, file=csv_file)
        part_df = pd.read_csv(csv_file, header=1)
    else:
        noniid_labeldir_part = MNISTPartitioner(dataset_train.targets, args.num_users, partition="noniid-labeldir",
                                                dir_alpha=args.dir_alpha,
                                                seed=args.seed)
        dict_users = noniid_labeldir_part.client_dict
        partition_report(dataset_train.targets, dict_users,
                         class_num=10,
                         verbose=False, file=csv_file)
        part_df = pd.read_csv(csv_file, header=1)
    # end
    frequence = [part_df.iloc[0, i] for i in range(1, 11)]
    validation_idx = mnist_one_client_from_designed_frequence(dataset_test, frequence, 1000)
    img_size = dataset_train[0][0].shape

    # draw distribution
    part_df = part_df.set_index('client')
    col_names = [f"class{i}" for i in range(10)]
    for col in col_names:
        part_df[col] = (part_df[col] * part_df['Amount']).astype(int)
    part_df[col_names].plot.barh(stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.savefig(f"noniid对验证集loss的影响_users{args.num_users}_seed{args.seed}_alpha{args.dir_alpha}_iid{args.iid}.svg",
                bbox_inches='tight')
    plt.close()
    print("draw end")
    # draw end

    # model
    len_in = 1
    for x in img_size:
        len_in *= x
    net = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)

    w_glob = net.state_dict()
    # 记录全局模型的loss的变化过程，用于绘图
    loss_train_list = [[] for i in range(args.num_users)]
    kind = 'layer_hidden.weight'
    print("Aggregation over all clients")
    # 全局weight，w_locals[idx]为idx的weight，只保存最近一轮的weight
    w_locals = [w_glob for i in range(args.num_users)]
    net.train()
    # 2、fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        for i in range(args.num_users):
            # 2.1、训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[i])
            w, loss = local.train(net=copy.deepcopy(net).to(args.device),
                                  data_poisoning_mp=None)
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            _, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w,
                                     dataset=dataset_test, args=args, idx=validation_idx)
            loss_train_list[i].append(round(test_loss, 5))
        w_glob = FedAvg(w_locals)
        # debug
        loss_last_list = [loss_train_list[i][-1] for i in range(len(loss_train_list))]
        print(f"epoch-{iter}, loss:{str(loss_last_list)}")
        net.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # 每一轮进行测试
        net.eval()
        # acc_test, loss_test, asr_test = mnist_test(net, dataset_test, args, [], None)
        net.train()

    filename = f"seed{args.seed}_alpha{args.dir_alpha}_iid{args.iid}_users{args.num_users}.txt"
    file = open(filename, mode='a')
    for loss_list in loss_train_list:
        for i, item in enumerate(loss_list):
            if i > 0:
                file.write(' ')
            file.write(str(item))
        file.write('\n')
    file.flush()
    file.close()
    # record_datalist(loss_train_list,
    #                 generate_name("", args.seed, args.num_users, args.num_attackers, args.frac, args.epochs,
    #                               args.data_poisoning,
    #                               args.model_poisoning, args.model, args.dataset, "loss", args.dir_alpha))
