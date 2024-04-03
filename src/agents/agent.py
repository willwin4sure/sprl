"""
agent.py

This module contains the Agent class, the abstract base class for all agents,
which take in a game and game state and output an action.
"""

from abc import ABC, abstractmethod

from src.games.game import Game, GameState


class Agent(ABC):
    """
    An abstract base class for all agents.
    """

    @abstractmethod
    def action(self, game: Game, state: GameState) -> int:
        """
        Outputs an action given a game and state.
        """
        raise NotImplementedError
