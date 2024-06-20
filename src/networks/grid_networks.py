from typing import Tuple

import torch

from src.networks.network import Network


class ResidualBlock(torch.nn.Module):
    """
    Single residual block consisting of two convolutional layers and batch normalization.
    """
    
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


class BasicGridNetwork(Network):
    """
    Residual tower for evaluating positions in grid-based games.
    """

    def __init__(self, num_rows, num_cols, action_size, history_size, num_blocks, num_channels, num_policy_channels=2, num_value_channels=1):
        super(BasicGridNetwork, self).__init__()

        self.conv = torch.nn.Conv2d(2 * history_size + 1, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_channels)

        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_blocks)
        ])

        self.policy_conv = torch.nn.Conv2d(num_channels, num_policy_channels, kernel_size=1)
        self.policy_size = num_policy_channels * num_rows * num_cols
        self.policy_fc = torch.nn.Linear(self.policy_size, action_size)

        self.value_conv = torch.nn.Conv2d(num_channels, num_value_channels, kernel_size=1)
        self.value_size = num_value_channels * num_rows * num_cols
        self.value_fc1 = torch.nn.Linear(self.value_size, num_channels)
        self.value_fc2 = torch.nn.Linear(num_channels, 1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network.

        Args:
            x (torch.Tensor): (B, C, H, W), as from embed

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (B, A), (B, 1)
        """
        x = torch.nn.functional.relu(self.bn(self.conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_conv(x))
        policy = policy.view(-1, self.policy_size)
        policy = self.policy_fc(policy)

        value = torch.nn.functional.relu(self.value_conv(x))
        value = value.view(-1, self.value_size)
        value = torch.nn.functional.relu(self.value_fc1(value))
        value = torch.nn.functional.tanh(self.value_fc2(value))

        return policy, value