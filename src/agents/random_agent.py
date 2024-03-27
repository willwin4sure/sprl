"""
random_agent.py

An agent that samples actions randomly.
"""

import numpy as np

from src.games.game import Game, GameState
from src.agents.agent import Agent


class RandomAgent(Agent):
    def action(self, game: Game, state: GameState) -> int:
        action_mask = game.action_mask(state)
        return np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
