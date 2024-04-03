"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Tuple

import numpy as np
import tqdm

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.uct.uct_node import UCTNode


def UCT_search(game: Game, game_state: GameState, policy: Policy, num_iters: int, c: float = 1.0, train: bool = True) -> Tuple[np.ndarray, float]:
    """
    Perform num_iters iterations of the UCT algorithm from the given game state
    using the exploration parameter c. Return the distribution of visits to each direct child.

    Requires that game_state is a non-terminal state.
    """

    # set root action to -1 so can identify it and add noise
    root = UCTNode(game, game_state, -1)

    for _ in range(num_iters):
        # for _ in tqdm.tqdm(range(num_iters)):
        leaf = root.select_leaf(c)
        if leaf.is_terminal:
            # compute the value estimate of the player at the terminal leaf
            value_estimate = game.rewards(leaf.game_state)[
                leaf.game_state.player]

        else:
            # run the neural network to get prior policy and value estimate of the player at the leaf
            child_priors, value_estimate = policy.action(game, leaf.game_state)

            # if leaf == root:
            #     print("Priors", child_priors)

            # expand the non-terminal leaf node
            leaf.expand(child_priors, train)

        # backup the value estimate along the path to the root
        leaf.backup(value_estimate)

    # print("Number visits", root.child_number_visits)
    # print("Q values", root.child_Q())
    # print("U values", root.child_U())

    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()]
