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


class PentagoNetwork(Network):
    """
    Residual tower and embed function for the game of Pentago. 
    """

    def __init__(self, num_blocks: int, num_channels: int):
        super(PentagoNetwork, self).__init__()

        self.conv = torch.nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_channels)

        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_blocks)
        ])

        self.policy_conv = torch.nn.Conv2d(num_channels, 8, kernel_size=1)
        self.policy_fc = torch.nn.Linear(8 * 6 * 6, 288)

        self.value_conv = torch.nn.Conv2d(num_channels, 4, kernel_size=1)
        self.value_fc1 = torch.nn.Linear(4 * 6 * 6, num_channels)
        self.value_fc2 = torch.nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network.

        Args:
            x (torch.Tensor): (B, C=3, H=6, W=6), as from embed

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, 288), (B, 1)
        """
        x = torch.nn.functional.relu(self.bn(self.conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_conv(x))
        policy = policy.view(-1, 8 * 6 * 6)
        policy = self.policy_fc(policy)

        value = torch.nn.functional.relu(self.value_conv(x))
        value = value.view(-1, 4 * 6 * 6)
        value = torch.nn.functional.relu(self.value_fc1(value))
        value = torch.nn.functional.tanh(self.value_fc2(value))

        return policy, value
    
    def embed(self, game: Game, state: GameState) -> torch.Tensor:
        """
        Embeds a single Pentago state into a tensor.
        
        The tensor has two channels of size 6 x 6, where the first
        channel represents a bitmask of the current player's stones,
        and the second channel represents a bitmask of the opponent's stones.
        It also has a third color channel representing the current player
        (1 for first player, 0 for second player).
        """
        player_stones = torch.tensor(state.board == state.player, dtype=torch.float32).view(1, 1, 6, 6)
        opponent_stones = torch.tensor(state.board == (1 - state.player), dtype=torch.float32).view(1, 1, 6, 6)
        color_channel = torch.tensor([1 - state.player for _ in range(6 * 6)]).view(1, 1, 6, 6)

        output = torch.cat([player_stones, opponent_stones, color_channel], dim=1)  # (1, 3, 6, 6)

        return output
