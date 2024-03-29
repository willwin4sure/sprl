from typing import Tuple
import torch

from src.games.game import Game, GameState


class Network(torch.nn.Module):
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single state into a tensor.
        """
        raise NotImplementedError


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return torch.nn.functional.relu(x + residual)


class ConnectFourNetwork(Network):
    """
    Basic convolutional neural network for default Connect Four parameters.
    """

    def __init__(self):
        super(ConnectFourNetwork, self).__init__()

        self.conv = torch.nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(64)

        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64, 64, kernel_size=3, stride=1, padding=1),
        ])

        self.policy_conv = torch.nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc1 = torch.nn.Linear(2 * 6 * 7, 64)
        self.policy_fc2 = torch.nn.Linear(64, 7)

        self.value_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = torch.nn.Linear(1 * 6 * 7, 64)
        self.value_fc2 = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.nn.functional.relu(self.bn(self.conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_conv(x))
        policy = policy.view(-1, 2 * 6 * 7)
        policy = torch.nn.functional.relu(self.policy_fc1(policy))
        policy = self.policy_fc2(policy)

        value = torch.nn.functional.relu(self.value_conv(x))
        value = value.view(-1, 1 * 6 * 7)
        value = torch.nn.functional.relu(self.value_fc1(value))
        value = torch.nn.functional.tanh(self.value_fc2(value))

        return policy, value
    
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single Connect Four state into a tensor. The tensor
        should have two channels of size 6 x 7, where the first
        channel represents a bitmask of the current player's stones,
        and the second channel represents a bitmask of the opponent's stones.
        """
        player_stones = torch.tensor(state.board == state.player, dtype=torch.float32).view(1, 1, 6, 7)
        opponent_stones = torch.tensor(state.board == 1 - state.player, dtype=torch.float32).view(1, 1, 6, 7)

        return torch.cat([player_stones, opponent_stones], dim=1)
