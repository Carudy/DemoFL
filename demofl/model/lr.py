import numpy as np
import torch

from .wrapper import ModelWrapper


# simple LR model
class LinearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_label):
        super(LinearRegression, self).__init__()
        self.l1 = torch.nn.Linear(n_feature, n_label)

    def forward(self, x):
        out = self.l1(x)
        return out


# simple LR modelwrapper
class LRWrapper(ModelWrapper):
    def __init__(self, n_feature=10, n_label=1):
        super(LRWrapper, self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label

        self.model = LinearRegression(n_feature, n_label)
        self.lr = 1e-2
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr)
        self.loss_func = torch.nn.MSELoss()
        self.epoch_max = 10
        self.get_ready()
        self.n_sample = 0

    def get_ready(self):
        # generate random datasets
        batch = 10
        _x = np.random.randn(batch, self.n_feature).astype(np.float32)
        _y = np.random.randn(batch).astype(np.float32)
        self.datasets = torch.utils.data.DataLoader(list(zip(_x, _y)))
        self.epoch_max = 5
        self.epoch = 0
        self.iter = iter(self.datasets)

    def train_epoch(self):
        if self.epoch < self.epoch_max:
            _x, _y = next(self.iter)
            output = self.model(_x)
            self.optimizer.zero_grad()
            loss = self.loss_func(output, _y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
        self.n_sample = 1
        self.epoch += 1

    def get_tensor_type(self):
        return 'torch'

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, params):
        self.model.load_state_dict(params)
