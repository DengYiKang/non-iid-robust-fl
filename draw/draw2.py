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


# target attack，隐瞒被毒害的标签信息，验证集失效


def draw_3_asr_acc(asr_list, acc_list, save_name):
    plt.figure()
    ax = plt.subplot(111)
    labels = ['normal client', 'malicious client']
    acc = [acc_list[1][-1], acc_list[0][-1]]
    asr = [asr_list[1][-1], asr_list[0][-1]]
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plot1 = ax.bar(x - width / 2, acc, width, label='Accuracy')
    plot2 = ax.bar(x + width / 2, asr, width, label='Attack Success Rate')
    for value in plot1:
        height = value.get_height()
        ax.text(value.get_x() + value.get_width() / 2.,
                1.002 * height, '%.1f' % height, ha='center', va='bottom')
    for value in plot2:
        height = value.get_height()
        ax.text(value.get_x() + value.get_width() / 2.,
                1.002 * height, '%.1f' % height, ha='center', va='bottom')
    ax.set_ylabel("Rate(%)")
    # x轴刻度标签位置不进行计算
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=False, ncol=5)
    plt.savefig(save_name)


def draw_2_loss(loss_list, save_name):
    plt.figure()
    x = np.linspace(1, 100, 100)
    colors = ['b', 'g']
    labels = ['malicious client', 'normal client']
    for i in range(len(loss_list)):
        plt.plot(x, loss_list[i], color=colors[i], linestyle='-', label=labels[i])
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Test Loss")
    plt.savefig(save_name)
    plt.close()


def draw_distribution(fre, size, save_name):
    plt.figure()
    cls = [i for i in range(10)]
    yticks = [500 * i for i in range(7)]
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
    # plt.yticks(yticks)
    plt.savefig(save_name)
    plt.close()


if __name__ == '__main__':
    args = args_parser()
    # seed 20 34 50 60 70
    # [7, 9] 3
    source_labels = [1]
    target_label = 2
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
    # 数据中毒的数据分布
    mp[0][1] = 0.5
    mp[0][2] = 0.2
    mp[0][3] = 0.2
    mp[0][6] = mp[0][7] = mp[0][8] = 0.1

    # 中毒节点上传的标签分布，标签为7的数量被更改为0，其他不变
    mp[1][1] = 0.0
    mp[1][2] = 0.2
    mp[1][3] = 0.2
    mp[1][6] = mp[1][7] = mp[1][8] = 0.1
    for i in range(2):
        for j in range(10):
            if j not in mp[i]:
                frequence[i].append(0.05)
            else:
                frequence[i].append(mp[i][j])
    # attack map
    attack_mp = {}
    for i in range(0, 10):
        if i in source_labels:
            attack_mp[i] = target_label
        else:
            attack_mp[i] = i
    # load mydataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    img_size = dataset_train[0][0].shape
    # build model
    len_in = 1
    for x in img_size:
        len_in *= x
    poisoning_net = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    normal_net = copy.deepcopy(poisoning_net)
    data_idx_test = mnist_one_client_from_designed_frequence(dataset_test, frequence[1], 1000)
    data_idx_train = [None, None]
    # 异常客户端，含有中毒标签
    data_idx_train[0] = mnist_one_client_from_designed_frequence(dataset_train, frequence[0], 6000)
    # 正常客户端
    data_idx_train[1] = mnist_one_client_from_designed_frequence(dataset_train, frequence[1], 6000)

    draw_distribution(frequence[0], 6000, "../imgs/异常客户端的数据分布.svg")
    draw_distribution(frequence[1], 6000, "../imgs/正常客户端的数据分布.svg")
    loss_locals = [[], []]
    asr_list = [[], []]
    acc_list = [[], []]
    # 异常客户端
    for iter in range(100):
        poisoning_local = LocalUpdate(args=args, dataset=dataset_train, idxs=data_idx_train[0], local_ep=1)
        w, loss = poisoning_local.train(net=copy.deepcopy(poisoning_net), data_poisoning_mp=attack_mp)
        poisoning_net.load_state_dict(w)
        _, test_loss = brca_test(net=copy.deepcopy(poisoning_net).to(args.device), w=w,
                                 dataset=dataset_test, args=args, idx=data_idx_test)
        loss_locals[0].append(test_loss)
        acc_test, loss_test, asr_test = mnist_test(poisoning_net, dataset_test, args, source_labels, target_label)
        asr_list[0].append(round(asr_test, 5))
        acc_list[0].append(round(acc_test.item(), 5))
        print(f"iter:{iter}, loss1={loss_locals[0][-1]}, asr1={asr_list[0][-1]}")

        normal_local = LocalUpdate(args=args, dataset=dataset_train, idxs=data_idx_train[0], local_ep=1)
        w, loss = normal_local.train(net=copy.deepcopy(normal_net), data_poisoning_mp=None)
        normal_net.load_state_dict(w)
        _, test_loss = brca_test(net=copy.deepcopy(normal_net).to(args.device), w=w,
                                 dataset=dataset_test, args=args, idx=data_idx_test)
        loss_locals[1].append(test_loss)
        acc_test, loss_test, asr_test = mnist_test(normal_net, dataset_test, args, source_labels, target_label)
        asr_list[1].append(round(float(asr_test), 5))
        acc_list[1].append(round(float(acc_test.item()), 5))
        print(f"iter:{iter}, loss2={loss_locals[1][-1]}, asr2={asr_list[1][-1]}")

    draw_2_loss(loss_locals, "../imgs/异常客户端与正常客户端之间loss的比较.svg")
    draw_3_asr_acc(asr_list, acc_list, "../imgs/异常客户端与正常客户端之间asr的比较.svg")
