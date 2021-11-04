import copy

import torch
import numpy as np
import random
from utils.options import args_parser
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar, AE
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
    # 1、各种参数
    # byzantine比例
    byzantine_proportion = 0.2
    # 两种分数的加权
    beta = 0.5
    # 初始化记录
    # loss_per_client[idx]为list，记录不同轮数时某个user的loss值
    # per_loss_per_client同理
    loss_per_client = {}
    per_loss_per_client = {}
    kind = 'layer_hidden.weight'
    for t in range(args.num_users):
        loss_per_client[t] = []
        per_loss_per_client[t] = []
    print("Aggregation over all clients")
    # 全局weight，w_locals[idx]为idx的weight，只保存最近一轮的weight
    w_locals = [w_glob for i in range(args.num_users)]
    # 个性化weight，与w_locals同理
    per_w_locals = [w_glob for i in range(args.num_users)]
    net.train()
    # 加载anomaly detection model以及优化器参数
    ae_model_path = './anomaly_detection/model/size_19360_batch_5_seed_10_loss_0.055.pth'
    ae_net = AE().to(args.device)
    optimizer = torch.optim.SGD(ae_net.parameters(), lr=0.001)
    checkpoint = torch.load(ae_model_path)
    ae_net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    rebuild_loss_func = torch.nn.MSELoss().to(args.device)
    # 2、fed训练
    for iter in range(args.epochs):
        # loss_locals为list，保存当前训练轮次中所有user的loss， per_loss_locals同理
        loss_locals = []
        per_loss_locals = []
        # anomaly detection score list
        e_scores = []
        # data validation score list
        f_scores = []
        # 分数高的idx集合
        trust_idxs = []
        m = max(args.num_users, 1)
        for idx in range(args.num_users):
            # 2.1、训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, p_w, loss, per_loss = local.train(net=copy.deepcopy(net).to(args.device),
                                                 per_net=copy.deepcopy(per_net_list[idx]).to(args.device))
            loss_per_client[idx].append(loss)
            per_loss_per_client[idx].append(per_loss)
            w_locals[idx] = copy.deepcopy(w)
            per_w_locals[idx] = copy.deepcopy(p_w)
            loss_locals.append(copy.deepcopy(loss))
            per_loss_locals.append(copy.deepcopy(per_loss))
            if idx < args.num_users * byzantine_proportion:
                # todo: add attack: w_locals[idx]=addAttack(w_locals[idx])
                pass
        # 2.2、anomaly detection
        for idx in range(args.num_users):
            origin = per_loss_locals[idx][kind].view(1, -1)
            encoded, decoded = ae_net(origin)
            loss = rebuild_loss_func(decoded, origin)
            e_scores[idx] = loss.data
        e_mean = np.mean(e_scores)
        e_std = np.std(e_scores)
        e_scores = [(item - e_mean) / e_std for item in e_scores]
        # 2.3、data validation
        # todo:目前只是把本地数据集的loss作为分数，未来需要将这里改成在共享数据集上的loss
        for idx in range(args.num_users):
            f_scores[idx] = per_loss_locals[idx]
        f_mean = np.mean(f_scores)
        f_std = np.std(f_scores)
        f_scores = [(item - f_mean) / f_std for item in f_scores]
        e_sum = np.sum(e_scores)
        f_sum = np.sum(f_scores)
        # 正则化
        for idx in range(args.num_users):
            e_scores[idx] /= e_sum
            f_scores[idx] /= f_sum
        scores = [beta * e_scores[idx] + (1 - beta) * f_scores[idx] for idx in range(args.num_users)]
        # 2.4、取分数高于mean的clients的数据，其他分数置为0
        score_mean = np.mean(scores)
        for idx in range(args.num_users):
            if score_mean > scores[idx]:
                scores[idx] = 0
            else:
                trust_idxs.append(idx)
        scores_sum = np.sum(scores)
        scores = [item / scores_sum for item in scores]
        # 2.5、将分数高的数据喂给ae net训练
        # 2.6、更新全局model
        pass
    pass
