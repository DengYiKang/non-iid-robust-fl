import torch
from torch.utils.data.dataset import Dataset, T_co


class AETrainDataSet(Dataset):

    def __init__(self, args, dataset) -> None:
        # self.data = torch.load('./model/data/encoder/train/size_32k_standard.pt')
        self.data = torch.load(dataset)
        self.args = args

    def __getitem__(self, index) -> T_co:
        return self.data[index].to(self.args.device)

    def __len__(self):
        return len(self.data)


class AETestDataSet(Dataset):

    def __init__(self, args, dataset) -> None:
        # self.data = torch.load('./model/data/encoder/test/size_8k_standard.pt')
        self.data = torch.load(dataset)
        self.args = args

    def __getitem__(self, index) -> T_co:
        return self.data[index].to(self.args.device)

    def __len__(self):
        return len(self.data)
