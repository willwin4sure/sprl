"""
self_play.py

This module contains the self-play function, which plays a game between two policies
and returns the game states and action distributions, as well as the final result.

It also contains a larger function which generates a dataset of self-play games.
"""

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.games.game import Game, GameState
from src.policies.policy import Policy


def self_play(game: Game, policies: Tuple[Policy, Policy]) -> Tuple[List[GameState], List[np.ndarray], float]:
    """
    Play a game between two policies and return the game states, action distributions, and final reward for player 0.
    """

    states: List[GameState] = []
    distributions: List[np.ndarray] = []

    state: GameState = game.start_state()

    move_count = 0

    while not game.is_terminal(state):
        states.extend(game.symmetrize_state(state, list(range(game.num_symmetries()))))

        policy = policies[state.player]

        distribution, _ = policy.action(game, state)
        
        if move_count >= 5:
            # decrease temperature for sampling when later on in the game
            distribution = distribution ** 10
            distribution /= np.sum(distribution)

        distributions.extend(game.symmetrize_action_distribution(distribution, list(range(game.num_symmetries()))))
        
        action = np.random.choice(len(distribution), p=distribution)
        state = game.next_state(state, action)

        move_count += 1

    return states, distributions, game.rewards(state)[0]


def run_iteration(game: Game, policies: Tuple[Policy, Policy], num_games: int) -> Tuple[List[GameState], List[np.ndarray], List[float]]:
    """
    Run a single iteration of self-play, generating a dataset of games.

    The dataset is a tuple of lists of states, distributions, and rewards.

    Each tuple (states[i], distributions[i], rewards[i]) is a datapoint
    that can be used to train a neural network directly.
    """

    all_states: List[GameState] = []
    all_distributions: List[np.ndarray] = []
    all_rewards: List[float] = []

    with tqdm(range(num_games)) as pbar:
        for _ in pbar:
            states, distributions, reward = self_play(game, policies)
            all_states.extend(states)
            all_distributions.extend(distributions)

            for state in states:
                if state.player == 0:
                    all_rewards.append(reward)
                else:
                    all_rewards.append(-reward)

            pbar.set_description(f"{len(all_states)} states generated")

    assert len(all_states) == len(all_distributions) == len(all_rewards)

    return all_states, all_distributions, all_rewards
