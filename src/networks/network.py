from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Network(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network. Returns a policy and a value.
        """
