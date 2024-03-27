"""
random_policy.py

This module contains the RandomPolicy class, a basic implementation
of a Policy class that just outputs a uniform action distribution.
"""

from typing import Tuple
import numpy as np
import torch

from src.games.game import Game, GameState
from src.policies.policy import Policy


class RandomPolicy(Policy):
    """
    A Policy that outputs a uniform action distribution.
    """

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a uniform action distribution and a value estimate given a game and state.
        """
        action_probs = game.action_mask(state) / np.sum(game.action_mask(state))
        return action_probs, 0.0
