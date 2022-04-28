from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch

from ..utils import *


class DataSpliter:
    def __init__(self, name='mnist'):
        if name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
            self.train_data = datasets.MNIST(root=BASE_PATH / "data", transform=transform, train=True, download=True)
            self.test_data = datasets.MNIST(root=BASE_PATH / "data", transform=transform, train=False, download=True)
            self.test_dataset = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=32, shuffle=True)
            self.size = 0.25

    def get_piece(self):
        train_indices, test_indices, _, _ = train_test_split(
            range(len(self.train_data)),
            self.train_data.targets,
            stratify=self.train_data.targets,
            test_size=self.size,
        )
        return Subset(self.train_data, test_indices)
