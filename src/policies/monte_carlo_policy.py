"""
monte_carlo_policy.py

This module contains the MonteCarloPolicy class, a basic implementation
of a Policy class that uses a monte carlo simulation.
"""

from typing import Tuple

import numpy as np

from src.games.game import Game, GameState
from src.policies.policy import Policy


class MonteCarloPolicy(Policy):
    """
    A policy that uses Monte Carlo rollouts to output a probability distribution over actions.
    """

    def __init__(self, temperature: float, num_simulations: int):
        """
        Initialize the MonteCarloPolicy with a temperature and number of simulations.

        Temperature is used to convert from the calculated utility of each action to a probability distribution, via
        the formula: P(a) = exp(Q(a) / T) / sum_a'(exp(Q(a') / T))

        num_simulations is the number of simulations to run for each action.
        """
        self.temperature = temperature
        self.num_simulations = num_simulations

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a probability distribution over valid actions and a value estimate given a game and state.
        """
        action_mask = game.action_mask(state)
        num_actions = action_mask.size
        legal_actions = np.sum(action_mask)

        # initialize Q and N for each action
        Q = np.zeros(num_actions)
        N = np.ones(num_actions)  # initialize to 1 to avoid division by zero

        who = state.player

        # run simulations
        action_idx = 0
        for i in range(legal_actions):
            while action_mask[action_idx] == 0:
                action_idx += 1

            iterations = (i < self.num_simulations % legal_actions) \
                + self.num_simulations // legal_actions

            for _ in range(iterations):
                next_state = game.next_state(state, action_idx)

                while not game.is_terminal(next_state):
                    # select a random action and play it with np.random.multinomial
                    next_action_mask = game.action_mask(next_state)
                    action_idx_next = np.random.choice(
                        num_actions, p=next_action_mask / np.sum(next_action_mask))
                    next_state = game.next_state(next_state, action_idx_next)

                # calculate the reward for the current player
                reward = game.rewards(next_state)[who]
                Q[action_idx] += reward

            # set, not +=, because we initialized to 1.
            N[action_idx] = iterations

        # calculate the utility of each action
        U = Q / N

        # also compute the total utility
        total_utility = np.sum(Q) / self.num_simulations

        # set mask to -inf
        U[action_mask == 0] = -np.inf

        # convert to a probability distribution
        if self.temperature == 0:
            # greedy action
            P = np.zeros(num_actions)
            P[np.argmax(U)] = 1.0

        else:
            P = np.exp(U / self.temperature)
            # np.exp(-np.inf) = 0.0, so we're good here.
            P /= np.sum(P)

        return P, total_utility
