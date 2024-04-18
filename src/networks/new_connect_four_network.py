from typing import Tuple
import torch

from src.games.game import Game, GameState

from src.networks.network import Network


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
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)) + residual)

        return x


class NewConnectFourNetwork(Network):
    """
    Basic convolutional neural network for default Connect Four parameters.
    """

    def __init__(self, num_blocks: int, num_channels: int):
        super(NewConnectFourNetwork, self).__init__()

        self.conv = torch.nn.Conv2d(2, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_channels)

        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_blocks)
        ])

        self.policy_conv = torch.nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_fc = torch.nn.Linear(2 * 6 * 7, 7)

        self.value_conv = torch.nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_fc1 = torch.nn.Linear(1 * 6 * 7, num_channels)
        self.value_fc2 = torch.nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network.

        Args:
            x (torch.Tensor): (B, C=2, H=6, W=7), as from embed

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, 7), (B, 1)
        """
        x = torch.nn.functional.relu(self.bn(self.conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_conv(x))
        policy = policy.view(-1, 2 * 6 * 7)
        policy = self.policy_fc(policy)

        value = torch.nn.functional.relu(self.value_conv(x))
        value = value.view(-1, 1 * 6 * 7)
        value = torch.nn.functional.relu(self.value_fc1(value))
        value = torch.nn.functional.tanh(self.value_fc2(value))

        return policy, value
    
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single Connect Four state into a tensor.
        
        The tensor has two channels of size 6 x 7, where the first
        channel represents a bitmask of the current player's stones,
        and the second channel represents a bitmask of the opponent's stones.
        """
        player_stones = torch.tensor(state.board == state.player, dtype=torch.float32).view(1, 1, 6, 7)
        opponent_stones = torch.tensor(state.board == (1 - state.player), dtype=torch.float32).view(1, 1, 6, 7)

        output = torch.cat([player_stones, opponent_stones], dim=1)  # (B=1, C=2, H=6, W=7)
        
        return output