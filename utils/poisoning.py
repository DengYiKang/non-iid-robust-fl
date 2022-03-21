import numpy as np
import torch


def add_attack(w, attack):
    """
    model poisoning
    :param w:
    :param attack: options: "same value", "sign flipping", "gaussian noisy"
    :return: w after attack
    """
    if attack == "same value":
        for k in w.keys():
            w[k][:] = 100
    elif attack == "sign flipping":
        for k in w.keys():
            w[k] = w[k] * -1
    elif attack == "gaussian noisy":
        for k in w.keys():
            mean = 0
            std = 1
            noisy = std * torch.randn(w[k].size()) + mean
            w[k] += noisy.cuda()
    return w
