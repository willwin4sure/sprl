"""
policy.py

This module contains the Policy class, the abstract base class for all policies,
which take in a game and game state and output a probability distribution
over valid actions as well as a value estimate.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.games.game import Game, GameState


class Policy(ABC):
    """
    An abstract base class for all policies.
    """

    @abstractmethod
    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a probability distribution over valid actions and a value estimate given a game and state.

        The value estimate given is *relative*, meaning it is more positive if the current player is winning.
        """
        raise NotImplementedError
