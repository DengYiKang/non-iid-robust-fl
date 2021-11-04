import copy

import torch
import numpy as np
import random
from utils.options import args_parser
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdateDitto
from models.Fed import FedAvg

if __name__ == "__main__":
    # seed
    seed = 34
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    # args, dataset
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load mydataset and split users
    if args.dataset == 'mnist':
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
    # 个性化net
    per_net_list = []
    # 初始化个性化weight列表
    for i in range(args.num_users):
        p_net = copy.deepcopy(net)
        p_net.train()
        per_net_list.append(p_net)
    # training
    net.train()
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    loss_per_client = {}
    per_loss_per_client = {}
    kind = 'layer_hidden.weight'

    for t in range(args.num_users):
        loss_per_client[t] = []
        per_loss_per_client[t] = []
    print("Aggregation over all clients")
    # 全局weight
    w_locals = [w_glob for i in range(args.num_users)]
    # 个性化weight
    per_w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        per_loss_locals = []
        m = max(args.num_users, 1)
        for idx in range(args.num_users):
            local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, p_w, loss, per_loss = local.train(net=copy.deepcopy(net).to(args.device),
                                                 per_net=copy.deepcopy(per_net_list[idx]).to(args.device))
            loss_per_client[idx].append(loss)
            per_loss_per_client[idx].append(per_loss)
            w_locals[idx] = copy.deepcopy(w)
            per_w_locals[idx] = copy.deepcopy(p_w)
            loss_locals.append(copy.deepcopy(loss))
            per_loss_locals.append(copy.deepcopy(per_loss))

        # 更新全局模型
        w_glob = FedAvg(w_locals)
        # 将参数复制到net上
        net.load_state_dict(w_glob)
        # 更新个性化模型
        for idx in range(len(per_net_list)):
            per_net_list[idx].load_state_dict(per_w_locals[idx])
        # 打印loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        per_loss_avg = sum(per_loss_locals) / len(per_loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Average Per Loss {:.3f}'.format(iter, loss_avg, per_loss_avg))
        loss_train.append(loss_avg)
