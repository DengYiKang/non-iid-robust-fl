import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import random
from utils.options import args_parser
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar, AE
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import brca_test, test_img
from utils.poisoning import add_attack

SAME_VALUE_ATTACK = "same value"
SIGN_FLIPPING_ATTACK = "sign flipping"
GAUSSIAN_NOISY_ATTACK = "gaussian noisy"

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
    # 定义所有user的类信息
    cls = []
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
            dict_users = mnist_noniid_designed(dataset_train, cls, 600)
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
    # byzantine比例
    byzantine_proportion = 0.2
    # 全局更新计算式子中的新旧权值
    alpha = 0.5
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
        # loss_locals为list，保存当前训练轮次中所有user的loss
        loss_locals = []
        # anomaly detection score list
        e_scores = []
        # data validation score list
        f_scores = []
        # 分数高的idx集合
        trust_idxs = []
        m = max(args.num_users, 1)
        for idx in range(args.num_users):
            # 2.1、训练，获得上传参数，如果是byzantine，那么修改参数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net).to(args.device))
            loss_per_client[idx].append(loss)
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            if idx < args.num_users * byzantine_proportion:
                # 序号前指定的比例为byzantine，将上传的w替换。model poisoning
                w_locals[idx] = add_attack(w_locals[idx], GAUSSIAN_NOISY_ATTACK)
        # 2.2、anomaly detection, e_scores
        ae_net.eval()
        for idx in range(args.num_users):
            origin = loss_locals[idx][kind].view(1, -1)
            encoded, decoded = ae_net(origin)
            loss = rebuild_loss_func(decoded, origin)
            e_scores[idx] = loss.data
        e_mean = np.mean(e_scores)
        e_std = np.std(e_scores)
        e_scores = [(item - e_mean) / e_std for item in e_scores]
        # 2.3、data validation, f_scores
        for i in range(args.num_users):
            f_scores[i] = brca_test(net=copy.deepcopy(net).to(args.device).load_state_dict(w_locals[i]),
                                    dataset=dataset_test, args=args, idx=shared_dataset_test_idx[i])
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
        ae_net.train()
        losses = []
        for idx in range(len(scores)):
            if scores[idx] > 0:
                item = loss_locals[idx][kind].to(args.device)
                encoded, decoded = ae_net(item)
                loss = rebuild_loss_func(decoded, item)
                loss.backward()
                optimizer.step()
                losses.append(loss.data)
        print('ae_net train loss: {:.3f}'.format(np.mean(losses[-1])))
        # 2.6、更新全局model
        w_glob = alpha * w_glob
        for i in range(args.num_users):
            w_glob += (1 - alpha) * scores[i] * w_locals[i]
        net.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    plt.show()
    # plt.savefig('./model/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net.eval()
    acc_train, loss_train = test_img(net, dataset_train, args)
    acc_test, loss_test = test_img(net, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
