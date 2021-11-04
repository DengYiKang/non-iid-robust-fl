import os

import numpy as np
import torch
from sklearn.cluster import KMeans
import copy  # 用于联邦学习全局模型的复制过程
from torchvision import datasets, transforms

from utils import preprocess
from utils.sampling import mnist_noniid_more_classes, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, AE

index = 0
ae_model_path = './anomaly_detection/model/size_19360_batch_5_seed_10_loss_0.055.pth'


def model_gen(args, dataset_train, cls_list):
    """
    生成模型
    :param args:
    :param dataset_train:
    :param cls_list:
    :return: dict_users为用户数据集的划分，字典，dict_users[idx]为训练数据的索引集合，net_glob为网络
    """
    img_size = dataset_train[0][0].shape
    if args.dataset == 'mnist':
        dict_users = mnist_noniid_more_classes(dataset_train, args.num_users, cls_list, args.train_size)
    elif args.dataset == 'cifar':
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
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
    return dict_users, net_glob


def do_train(args, dataset_train, dict_users, net_glob):
    """
    训练
    :param args:
    :param dataset_train:
    :param dict_users:
    :param net_glob:
    :return:
    """
    print(net_glob)
    # training
    net_glob.train()
    loss_per_client = {}
    w_locals = []
    kind = 'layer_hidden.weight'

    for t in range(args.num_users):
        loss_per_client[t] = []
    print("Aggregation over all clients")
    idxs_users = range(args.num_users)
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        loss_per_client[idx].append(loss)
        w_locals.append(copy.deepcopy(w)[kind].view(1, -1))
        print('round {:3d} client {:3d}, loss {:.3f}'.format(args.index, idx, loss))
    # ae_net = torch.load(ae_model_path).cuda()
    checkpoint = torch.load(ae_model_path)
    ae_net = AE().cuda()
    ae_net.load_state_dict(checkpoint['net'])
    features = []
    for w in w_locals:
        encoded, decoded = ae_net(w.cuda())
        features.append(encoded)
    torch.save(features, './save/data/kmeans/input/' + str(args.index) + '.pt')
    args.index += 1


def get_dataset(args):
    """
    获取训练集
    :param args:
    :return:
    """
    dataset_train = None
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
    return dataset_train


def data_gen(args, cls_list, dataset_train):
    dict_users, net_glob = model_gen(args, dataset_train, cls_list)
    do_train(args, dataset_train, dict_users, net_glob)


def union(name):
    """
    将小文件合并
    :param name:
    :return:
    """
    files = preprocess.walkFile('./save/data/kmeans/input/')
    data = []
    for f in files:
        data.extend(torch.load(f))
    torch.save(data, './save/data/kmeans/tot/' + name + '.pt')


def clean(path):
    for root, dirs, files in os.walk(path):
        # 遍历文件
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                os.remove(file_path)


def make_clusters(k, name):
    """
    聚类
    :param k: k个簇
    :param name:
    :return:
    """
    data = torch.load('./save/data/kmeans/tot/' + name + '.pt')
    for i in range(len(data)):
        data[i] = data[i].cpu().tolist()
    t = [token for term in data for token in term]
    X = np.array(t)
    kmeans = KMeans(n_clusters=k).fit(X)
    print(kmeans.labels_)
    clean('./save/data/kmeans/input/')
    clean('./save/data/kmeans/tot/')


if __name__ == '__main__':
    # data = torch.load('./model/data/kmeans/input')
    args = args_parser()
    args.local_bs = 32
    args.local_ep = 50
    args.epochs = 1
    args.num_users = 8
    # args.classes=[[1, 2], [3, 4]]表示分为两大客户端群，一个拥有1、2两个类的数据，另一个拥有3、4两个类的数据
    args.classes = [[1, 2, 3], [2, 3, 4, 5], [6], [1], [9, 0]]
    args.index = 1
    args.train_size = 600
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train = get_dataset(args)
    for cls_list in args.classes:
        data_gen(args, cls_list, dataset_train)
    union('mess')
    make_clusters(len(args.classes), 'mess')
