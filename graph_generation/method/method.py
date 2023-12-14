from abc import ABC, abstractmethod

import torch as th
from torch.nn import Module


class Method(ABC):
    """Interface for graph generation methods."""

    def __init__(self, diffusion):
        self.diffusion = diffusion

    @abstractmethod
    def sample_graphs(self, target_size: th.Tensor, model: Module, sign_net: Module):
        pass

    @abstractmethod
    def get_loss(self, batch, model: Module, sign_net: Module):
        pass

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        self.diffusion.to(device)
        return self
