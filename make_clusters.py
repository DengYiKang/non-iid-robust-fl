import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy  # 用于联邦学习全局模型的复制过程
from torchvision import datasets, transforms

import preprocess
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_only_one_class
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar


def one_class_data_gen(args, dataset_train):
    img_size = dataset_train[0][0].shape
    if args.dataset == 'mnist':
        dict_users = mnist_noniid_only_one_class(dataset_train, args.num_users, args.class_idx)
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
    ae_net = torch.load('./save/data/encoder/model/size_32000_loss_0.23777720568701624.pkl').cuda()
    features = []
    for w in w_locals:
        encoded, decoded = ae_net(w.cuda())
        features.append(encoded)
    torch.save(features, './save/data/kmeans/input/' + str(args.class_idx) + '.pt')


def data_gen(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized mydataset')
    for i in range(10):
        args.class_idx = i
        one_class_data_gen(args, dataset_train)


def union():
    files = preprocess.walkFile('./save/data/kmeans/input/')
    data = []
    for f in files:
        data.extend(torch.load(f))
    torch.save(data, './save/data/kmeans/tot/8_per_class.pt')


def make_clusters():
    data = torch.load('./save/data/kmeans/tot/8_per_class.pt')
    for i in range(len(data)):
        data[i] = data[i].cpu().tolist()
    t = [token for term in data for token in term]
    X = np.array(t)
    kmeans = KMeans(n_clusters=10).fit(X)
    print(kmeans.labels_)


if __name__ == '__main__':
    # data = torch.load('./save/data/kmeans/input')
    args = args_parser()
    args.local_bs = 32
    args.local_ep = 50
    args.epochs = 1
    args.num_users = 8
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # data_gen(args)
    # union()
    make_clusters()
