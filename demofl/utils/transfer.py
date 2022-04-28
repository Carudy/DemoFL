import numpy as np


def tensor2bytes(x, tensor_type='numpy'):
    if tensor_type == 'numpy':
        return x.tobytes()
    elif tensor_type == 'torch':
        return x.float().numpy().tobytes()


def bytes2tensor(x, tensor_type='numpy'):
    """
        x: bytes
    """
    if tensor_type == 'numpy':
        return np.frombuffer(x, dtype=np.float32)
    elif tensor_type == 'torch':
        import torch
        np_arr = np.frombuffer(x, dtype=np.float32)
        return torch.tensor(np_arr)
