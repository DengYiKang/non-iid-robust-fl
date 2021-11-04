import copy
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mydataset import AE_DATASET
from utils.options import args_parser
from models.Nets import AE

if __name__ == '__main__':

    seed = 34
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train = AE_DATASET.AETrainDataSet(args, './anomaly_detection/data/train/size24k_standard.pt')
    dataset_test = AE_DATASET.AETestDataSet(args, './anomaly_detection/data/test/size6k_standard.pt')
    train_loader = DataLoader(dataset_train, batch_size=5, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)
    ae_net = AE().to(args.device)
    # train
    ae_net.train()
    optimizer = torch.optim.SGD(ae_net.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = torch.nn.MSELoss().to(args.device)
    for epoch in range(20):
        for batch_idx, item in enumerate(train_loader):
            optimizer.zero_grad()
            encoded, decoded = ae_net(item)
            loss = loss_func(decoded, item)
            loss.backward()
            optimizer.step()
        print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)
    # test
    ae_net.eval()
    test_loss = 0
    for idx, item in enumerate(test_loader):
        encoded, decoded = ae_net(item)
        tmp = F.mse_loss(decoded, item, reduction='mean').item()
        print('Test ', idx, '|', 'test loss:%.4f' % tmp)
        test_loss += tmp
    test_loss /= len(test_loader)
    print('average loss:%.4f' % test_loss)
    # torch.save(ae_net, './model/data/encoder/model/size_{}_loss_{}.pkl'.format(len(dataset_train), test_loss))
