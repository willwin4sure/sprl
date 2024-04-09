from abc import ABC, abstractmethod
from typing import Tuple
import torch

from src.games.game import Game, GameState


class Network(torch.nn.Module, ABC):
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network. Returns a policy and a value.
        """
    
    @abstractmethod
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single state into a tensor.
        """
        raise NotImplementedError

