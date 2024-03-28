"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Tuple
import numpy as np

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.uct.uct_node import UCTNode


def UCT_search(game: Game, game_state: GameState, policy: Policy, num_iters: int, c: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Perform num_iters iterations of the UCT algorithm from the given game state
    using the exploration parameter c. Return the distribution of visits to each direct child.

    Requires that game_state is a non-terminal state.
    """
    
    root = UCTNode(game, game_state, -1)  # don't need to remember the action we took into the root

    for _ in range(num_iters):
        leaf = root.select_leaf(c)
        if leaf.is_terminal:
            # compute the value estimate of the player at the terminal leaf
            value_estimate = game.rewards(leaf.game_state)[leaf.game_state.player]

        else:
            # run the neural network to get prior policy and value estimate of the player at the leaf
            child_priors, value_estimate = policy.action(game, leaf.game_state)

            # expand the non-terminal leaf node
            leaf.expand(child_priors)
        
        # backup the value estimate along the path to the root
        leaf.backup(value_estimate)

    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()]
