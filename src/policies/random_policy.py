"""
random_policy.py

This module contains the RandomPolicy class, a basic implementation
of a Policy class that just outputs a uniform action distribution
and a value estimate of 0.
"""

from typing import Tuple
import numpy as np

from src.games.game import Game, GameState
from src.policies.policy import Policy


class RandomPolicy(Policy):
    """
    A Policy that outputs a uniform action distribution and a value estimate of 0.
    """

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a uniform action distribution and a value estimate of 0.
        """
        action_mask = game.action_mask(state)
        action_probs = action_mask / np.sum(action_mask)
        return action_probs, 0.0
