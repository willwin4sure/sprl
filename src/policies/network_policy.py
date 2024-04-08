"""
network_policy.py

This module contains the NetworkPolicy class, a basic implementation
of a Policy class that uses a neural network.
"""

from typing import Tuple
import numpy as np
import torch

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.networks.network import Network


class NetworkPolicy(Policy):
    """
    A Policy that uses a neural network to output a probability distribution over actions.
    """

    def __init__(self, network: Network, symmetrize: bool = False):
        self.network = network
        self.symmetrize = symmetrize

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a probability distribution over valid actions and a value estimate given a game and state.
        """
        if self.symmetrize:
            symmetry = np.random.randint(game.num_symmetries())
            state = game.symmetrize_state(state, [symmetry])[0]

        input = self.network.embed(game, state)  # (B=1, C=2, H=6, W=7)
        policy, value = self.network(input)  # (B=1, A=7), (B=1, V=1)

        # the network outputs logits, so we need to apply softmax
        policy = torch.nn.functional.softmax(policy, dim=1)
        policy = policy.squeeze().detach().numpy()  # (A=7)
        value = value.squeeze().detach().numpy()  # (V=1)

        policy *= game.action_mask(state)  # mask out illegal actions
        if np.sum(policy) == 0:
            # if policy is too small, output uniform distribution
            policy = game.action_mask(state)

        policy /= np.sum(policy)

        if self.symmetrize:
            policy = game.symmetrize_action_distribution(policy, [game.inverse_symmetry(symmetry)])[0]

        return policy, value
