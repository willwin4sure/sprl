import torch

from src.games.game import Game, GameState


class Network(torch.nn.Module):
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single state into a tensor.
        """
        raise NotImplementedError

