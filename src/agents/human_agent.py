"""
human_agent.py

An agent that allows a human to play a game interactively.
"""

import numpy as np

from src.games.game import Game, GameState
from src.agents.agent import Agent


def get_human_action(action_mask: np.ndarray) -> int:
    """
    Get a human action from the command line.
    """
    while True:
        try:
            action = int(input("Enter an action: "))
            if 0 <= action < len(action_mask) and action_mask[action]:
                return action
            else:
                print("Invalid action. Please enter a legal action.")

        except ValueError:
            print("Invalid input. Please enter an integer.")


class HumanAgent(Agent):
    """
    An agent that allows a human to play a game interactively.
    """

    def action(self, game: Game, state: GameState) -> int:
        """
        Outputs an action given a game and state, by asking the human to input an action.
        """
        action_mask = game.action_mask(state)
        return get_human_action(action_mask)
