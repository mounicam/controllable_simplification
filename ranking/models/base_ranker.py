import torch
import numpy as np
from torch.autograd import Variable
from gaussian_binner import GaussianBinner


class BaseRanker:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.model = None
        self.lr = lr

        self.binner = GaussianBinner()

    def set_model(self, d_in, dropout=0.2):
        h, d_out = 100, 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_in, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, d_out)
        )

    def set_device(self, device):
        self.model.to(device)

    def predict(self, test_x, test_segs):
        test_x = self.binner.transform(np.array(test_x))
        test_x = Variable(torch.FloatTensor(test_x))
        return [score[0] for score in self.model(test_x).data.numpy()], test_segs
