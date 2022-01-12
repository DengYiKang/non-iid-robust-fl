import copy

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import hdmedians as hd


def trimmedMean(w_locals, args, kind, alpha):
    """
    trimmed mean
    :param w_locals:
    :param args:
    :param kind:
    :param alpha:
    :return:
    """
    w_kind_locals = [w_locals[idx][kind].view(1, -1) for idx in range(args.num_users)]
    w_kind_norm_locals = [w_kind_locals[idx].pow(2).sum().sqrt().item() for idx in range(args.num_users)]
    drop_nodes = np.union1d(np.argsort(w_kind_norm_locals)[:int(alpha * args.num_users)],
                            np.argsort(w_kind_norm_locals)[::-1][:int(alpha * args.num_users)])
    sc = [{True: 0, False: 1}[idx in drop_nodes] for idx in range(args.num_users)]
    sc_sum = np.sum(sc)
    sc = [sc[idx] / sc_sum for idx in range(args.num_users)]
    w_glob = copy.deepcopy(w_locals[0])
    for k in w_glob.keys():
        w_glob[k] = 0
    for i in range(args.num_users):
        for k in w_glob.keys():
            w_glob[k] += sc[i] * w_locals[i][k]
    return w_glob


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
    score = [None for _ in range(args.num_users)]
    for idx in range(args.num_users):
        sorted_sims = np.sort(sims[idx])
        score[idx] = np.sum(sorted_sims[:closest_num + 1])
    selected_idx = np.argsort(score)[0]
    return w_locals[selected_idx]


def geoMed(w_locals, args, kind, groups):
    """
    Geometric Median
    :param w_locals:
    :param args:
    :param kind:
    :param groups:
    :return:
    """
    num_per_group = int(len(w_locals) / groups)
    w_group_avg = []
    idx_shard = [i for i in range(len(w_locals))]
    for i in range(groups):
        if i == groups - 1:
            rand_set = set(idx_shard)
        else:
            rand_set = set(np.random.choice(idx_shard, num_per_group, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        w_tmp = copy.deepcopy(w_locals[0])
        for k in w_tmp.keys():
            w_tmp[k] = 0
        for j in rand_set:
            for k in w_tmp.keys():
                w_tmp[k] += w_locals[j][k]
        for k in w_tmp.keys():
            w_tmp[k] /= len(rand_set)
        w_group_avg.append(w_tmp)
    w_glob = copy.deepcopy(w_locals[0])
    for k in w_locals[0].keys():
        points = []
        for i in range(len(w_group_avg)):
            points.append(w_group_avg[i][k].view(1, -1).tolist()[0])
        points_np = np.array(points)
        geo_med_point_np = np.array(hd.geomedian(points_np, axis=0))
        geo_med_point_torch = torch.from_numpy(geo_med_point_np)
        w_glob[k] = geo_med_point_torch.view(w_glob[k].shape)
    return w_glob
