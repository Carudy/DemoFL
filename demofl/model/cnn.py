import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .wrapper import ModelWrapper


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


class CNNWrapper(ModelWrapper):
    def __init__(self, name, dataset):
        super(CNNWrapper, self).__init__()
        self.name = name
        self.model = CNN()
        self.lr = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_datasets = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
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
        return self.model.state_dict()

    def set_parameters(self, params):
        self.model.load_state_dict(params)
