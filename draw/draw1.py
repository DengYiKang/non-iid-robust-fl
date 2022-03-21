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
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_designed, mnist_iid_duplicate, \
    random_select_on_dict_users, mnist_one_label_select, mnist_one_client_from_designed_frequence
from models.Update import LocalUpdate, RebalanceUpdate
from models.test import brca_test, test_img, mnist_test, mnist_all_labels_test


# 分布信息与实际不符将导致loss变高


def draw_2_loss(loss_list):
    plt.figure()
    x = np.linspace(1, 100, 100)
    colors = ['b', 'g']
    labels = ['loss1', 'loss2']
    for i in range(len(loss_list)):
        plt.plot(x, loss_list[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("test loss")
    plt.savefig(f"../imgs/分布信息与实际不符将导致loss高.svg")


def draw_distribution(fre, size):
    cls = [i for i in range(10)]
    num = [fre[i] * size for i in range(10)]
    plot = plt.bar(cls, num)
    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()
        if height == 0: continue
        plt.text(value.get_x() + value.get_width() / 2.,
                 1.002 * height, '%d' % int(height), ha='center', va='bottom')
    # Add labels and title
    plt.xlabel("class")
    plt.ylabel("num")
    plt.xticks(cls)
    plt.savefig(f"../imgs/客户端a的数据分布.png", dpi=600)
    # plt.savefig(f"imgs/客户端a的数据分布.svg")


if __name__ == '__main__':
    args = args_parser()
    # seed 20 34 50 60 70
    # [7, 9] 3
    source_labels = [7]
    target_label = 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    # args, dataset
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # data idx
    frequence = [[], []]
    mp = [{}, {}]
    mp[0][1] = 0.2
    mp[0][2] = 0.3
    mp[0][3] = 0.2
    mp[0][6] = mp[0][7] = mp[0][8] = 0.1
    mp[1][3] = 0.2
    mp[1][5] = 0.3
    mp[1][9] = 0.2
    mp[1][1] = mp[1][4] = mp[1][7] = 0.1
    for i in range(2):
        for j in range(10):
            if j not in mp[i]:
                frequence[i].append(0.01)
            else:
                frequence[i].append(mp[i][j])
    # load mydataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    img_size = dataset_train[0][0].shape
    # build model
    len_in = 1
    for x in img_size:
        len_in *= x
    net = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    data_idx_test = []
    for i in range(2):
        data_idx_test.append(mnist_one_client_from_designed_frequence(dataset_test, frequence[i], 1000))
    data_idx_train = mnist_one_client_from_designed_frequence(dataset_train, frequence[0], 6000)

    draw_distribution(frequence[0], 6000)

    loss_locals = [[], []]
    for iter in range(100):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=data_idx_train, local_ep=1)
        w, loss = local.train(net=copy.deepcopy(net), data_poisoning_mp=None)
        net.load_state_dict(w)
        acc, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w,
                                   dataset=dataset_test, args=args, idx=data_idx_test[0])
        print(f"iter:{iter}, loss1={test_loss}")
        loss_locals[0].append(test_loss)
        acc, test_loss = brca_test(net=copy.deepcopy(net).to(args.device), w=w,
                                   dataset=dataset_test, args=args, idx=data_idx_test[1])
        print(f"iter:{iter}, loss2={test_loss}")
        loss_locals[1].append(test_loss)

    # draw_2_loss(loss_locals)
