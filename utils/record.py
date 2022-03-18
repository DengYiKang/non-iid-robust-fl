import copy

import torch


def generate_name(prefix, seed, user, attackers, frac, epochs, dp, mp, model, dataset, dir_name, alpha):
    """
    根据参数返回唯一id
    :param seed:
    :param user:
    :param attackers:
    :param frac:
    :param epochs:
    :param dp:
    :param mp:
    :param model:
    :param dataset:
    :param dir_name:目录
    :param alpha
    :return:
    """
    name = "new_result/" + str(dir_name) + "/" + str(prefix) + "seed" + str(seed) + "_user" + str(
        user) + "_attackers" + str(attackers) + "_frac" + str(
        frac) + "_epochs" + str(
        epochs) + "_dp" + str(
        dp) + "_mp" + str(mp) + "_model" + str(model) + "_dataset" + str(dataset) + "_alpha" + str(alpha) + ".txt"
    return name


def generate_name_loss(seed, user, frac, epochs, dp, mp, iid):
    """
    根据参数返回唯一id，作为记录loss变化的文件名
    :param seed:
    :param user:
    :param frac:
    :param epochs:
    :param dp:
    :param mp:
    :param iid:
    :return:
    """
    name = "result/loss/seed" + str(seed) + "_user" + str(user) + "_frac" + str(frac) + "_epochs" + str(
        epochs) + "_dp" + str(
        dp) + "_mp" + str(mp) + "_iid" + str(iid) + ".txt"
    return name


def record_datalist(datalist, filename):
    """
    将datalist记录下来
    :param filename:
    :param datalist:
    :return:
    """
    file = open(filename, mode='a')
    for i, item in enumerate(datalist):
        if i > 0:
            file.write(' ')
        file.write(str(item))
    file.write('\n')
    file.flush()
    file.close()


def read_loss_record(file_name):
    """
    读取filename中的loss变化数据，该文件中，一行为一次实验的loss变化值，可以有多行，返回data_list，len(data_list)为行数
    :param fileName:
    :return:
    """
    file = open(file_name, mode='r')
    data_list = []
    while True:
        line = file.readline().split('\n')[0]
        if line:
            data = line.split(' ')
            data = [float(item) for item in data]
            data_list.append(data)
        else:
            break
    return data_list


if __name__ == '__main__':
    pass
