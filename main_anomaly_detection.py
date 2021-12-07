from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

import utils
import numpy as np
import random
import torch.nn.functional as F
from anomaly_detection import train_data_gen
from models.Nets import AE
from mydataset import AE_DATASET
from utils.options import args_parser
from utils.preprocess import preprocess


def data_gen(args):
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train_data_gen.data_gen(args, './anomaly_detection/data/input/')


def do_preprocess():
    input_path = 'anomaly_detection/data/input/'
    tot_path = 'anomaly_detection/data/tot/mnist_cnn_size30k.pt'
    train_path = 'anomaly_detection/data/train/mnist_cnn_size24k.pt'
    test_path = 'anomaly_detection/data/test/mnist_cnn_size6k.pt'
    train_standard_path = 'anomaly_detection/data/train/mnist_cnn_size24k_standard.pt'
    test_standard_path = 'anomaly_detection/data/test/mnist_cnn_size6k_standard.pt'
    mean, std = preprocess(input_path, tot_path, train_path, test_path, train_standard_path, test_standard_path)
    print("mean={}\tstd={}".format(mean, std))


if __name__ == '__main__':
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
    args.train_size = 600
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # data_gen(args)
    do_preprocess()
    dataset_train = AE_DATASET.AETrainDataSet(args, './anomaly_detection/data/train/mnist_cnn_size24k_standard.pt')
    dataset_test = AE_DATASET.AETestDataSet(args, './anomaly_detection/data/test/mnist_cnn_size6k_standard.pt')
    batch_size = 5
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)
    ae_net = AE().to(args.device)
    # train
    ae_net.train()
    optimizer = torch.optim.SGD(ae_net.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = torch.nn.MSELoss().to(args.device)
    max_epochs = 100
    for epoch in range(max_epochs):
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
    state_dict = {"net": ae_net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": max_epochs}
    # torch.save(ae_net, './anomaly_detection/model/size_{}_loss_{}.pkl'.format(len(dataset_train), test_loss))
    torch.save(state_dict,
               './anomaly_detection/model/mnist_cnn_dimIn500_size{}_batch{}_seed{}_loss{:.3f}.pth'.format(len(dataset_train),
                                                                                           batch_size, seed,
                                                                                           test_loss))
