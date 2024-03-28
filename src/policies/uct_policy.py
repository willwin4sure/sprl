"""
uct_policy.py

This is a policy that uses the UCT algorithm to select actions.
"""

from typing import Tuple
import numpy as np

from src.games.game import Game, GameState

from src.policies.policy import Policy

from src.uct.uct_alg import UCT_search

class UCTPolicy(Policy):
    """
    A policy that uses the UCT algorithm to select actions.
    """

    def __init__(self, policy: Policy, num_iters=1000, c: float = 1.0):
        self.policy = policy
        self.num_iters = num_iters
        self.c = c

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Select an action using the UCT algorithm.
        """
        return UCT_search(game, state, self.policy, self.num_iters, self.c)
