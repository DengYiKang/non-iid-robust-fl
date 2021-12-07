import copy

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def krum(w_locals, args, kind, byzantine_proportion):
    """
    krum算法
    :param w_locals: 各个client的模型参数
    :param args:
    :param kind: key，根据这个key计算距离
    :param byzantine_proportion: byzantine比例
    :return: 最终的全局模型参数
    """
    closest_num = max(int(args.num_users * (1 - byzantine_proportion)) - 2, 1)
    w_kind_locals = [w_locals[idx][kind].view(1, -1) for idx in range(args.num_users)]
    sims = [[None for item in range(args.num_users)] for _ in range(args.num_users)]
    for x in range(args.num_users):
        for y in range(args.num_users):
            if x == y:
                sims[x][y] = 0
            else:
                dist = F.pairwise_distance(w_kind_locals[x], w_kind_locals[y], p=2)
                sims[x][y] = dist.item()
    # krum距离，某个client到距离它最近的n-f-2个client的距离总和，f为byzantine数量
    score = []
    for idx in range(args.num_users):
        sorted_sims = np.sort(sims[idx])
        score[idx] = np.sum(sorted_sims[:closest_num + 1])
    selected_idx = np.argsort(score)[0]
    return w_locals[selected_idx]


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def geoMed(w_locals, args):
    """
    计算w_locals的geometric median作为global w
    :param w_locals:
    :param args:
    :return:
    """
    pass


def trimmedMean(w_locals, args, beta):
    """
    计算w_locals的trimmed mean
    :param w_locals:
    :param args:
    :param beta: 去掉头尾部分的比例
    :return:
    """
    pass
