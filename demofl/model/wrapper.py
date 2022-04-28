from abc import ABC, abstractmethod
import torch


class ModelWrapper(ABC):
    def __init__(self):
        self.epoch = 0

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def get_parameters(self):
        """
            The parameters should be dict format like {k: v}.
        """
        pass

    @abstractmethod
    def set_parameters(self, params):
        """
            "params" is dict formated like {k: v} where v's type is the same as "get_tensor_type()"
        """
        pass

    @abstractmethod
    def get_tensor_type(self):
        """
            The type that "get_parameters" method will return
            Usually be one of ['torch', 'numpy', 'tf']
        """
        pass
