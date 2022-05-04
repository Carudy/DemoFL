import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F

from .wrapper import ModelWrapper


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNetWrapper(ModelWrapper):
    def __init__(self, name, dataset):
        super(LeNetWrapper, self).__init__()
        self.name = name
        self.model = LeNet()
        self.lr = 1e-1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=5e-4)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_datasets = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        self.epoch = 0
        self.n_sample = 0
        self.tot_sample = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)
        self.epoch += 1
        for e in range(1):
            for (x_train, y_train) in tqdm(self.train_datasets, desc=f'Node {self.name}: '):
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                outputs = self.model(x_train)
                _, pred = torch.max(outputs.data, 1)
                self.optimizer.zero_grad()
                loss = self.loss_func(outputs, y_train)
                loss.backward()
                self.optimizer.step()
        self.n_sample = len(self.train_datasets)

    # def test_accuracy(self):
    #     self.test_correct = 0
    #     i = 0
    #     for (x_test, y_test) in self.test_datasets:
    #         outputs = self.model(x_test)
    #         _, pred = torch.max(outputs, 1)
    #         self.test_correct += torch.sum(pred == y_test.data)
    #         i += 1
    #         if i == 100:
    #             break

    def get_tensor_type(self):
        return 'torch'

    def get_parameters(self):
        return self.model.state_dict().copy()

    def set_parameters(self, params):
        self.model.load_state_dict(params)
