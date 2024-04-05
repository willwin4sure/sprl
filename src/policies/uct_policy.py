"""
uct_policy.py

This is a policy improvement operator that uses the UCT algorithm
combined with an existing policy to select actions.
"""

from typing import Tuple

import numpy as np

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.uct.uct_alg import uct_search


class UCTPolicy(Policy):
    """
    A policy that uses the UCT algorithm to select actions.
    """

    def __init__(self, policy: Policy, num_iters=1000, c: float = 1.0, train: bool = True, init_type: str = "parent"):
        """
        Initialize the UCTPolicy.

        init_type should be in ["zero", "parent"]
        """
        self.policy = policy
        self.num_iters = num_iters
        self.c = c  # exploration parameter
        self.train = train  # whether we are training or inference
        self.init_type = init_type

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Selects an action using the UCT algorithm.
        """
        return uct_search(game, state, self.policy, self.num_iters, self.c, self.train, self.init_type)
