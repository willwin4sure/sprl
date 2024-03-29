"""
self_play.py

This module contains the self-play function, which plays a game between two policies
and returns the game states and action distributions, as well as the final result.
It also contains a larger function which generates a dataset of self-play games.
"""

from typing import Tuple, List
import numpy as np
import torch

from tqdm import tqdm

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.networks.network import Network

def self_play(game: Game, policies: Tuple[Policy, Policy]) -> Tuple[List[GameState], List[np.ndarray], float]:
    """
    Play a game between two policies and return the game states, action distributions, and reward for player 0.
    """

    states: List[GameState] = []
    distributions: List[np.ndarray] = []

    state: GameState = game.start_state()

    moves = 0

    while not game.is_terminal(state):
        states.append(state)

        policy = policies[state.player]
        distribution, _ = policy.action(game, state)

        if moves >= 10:
            # decrease temperature when later on in the game
            distribution = distribution ** 10
            distribution /= np.sum(distribution)

        distributions.append(distribution)

        action = np.random.choice(len(distribution), p=distribution)
        state = game.next_state(state, action)

        moves += 1

    return states, distributions, game.rewards(state)[0]


def run_iteration(game: Game, policies: Tuple[Policy, Policy], num_games: int) -> Tuple[List[GameState], List[np.ndarray], List[float]]:
    """
    Run a single iteration of self-play, generating a dataset of games.
    """

    all_states: List[GameState] = []
    all_distributions: List[np.ndarray] = []
    all_rewards: List[float] = []

    with tqdm(range(num_games)) as pbar:
        for _ in pbar:
            states, distributions, reward = self_play(game, policies)
            all_states.extend(states)
            all_distributions.extend(distributions)
            all_rewards.extend([reward] * len(states))

            pbar.set_description(f"{len(all_states)} states generated")

    return all_states, all_distributions, all_rewards
