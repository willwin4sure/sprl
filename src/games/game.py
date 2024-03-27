"""
game.py

This module contains the Game class, the base class for any two-player,
zero-sum, perfect information, abstract strategy game.
"""

from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GameState:
    """
    Immutable base class to store the state of a game.
    """
    _board: np.ndarray = None
    player: int = 0
    winner: int = -1

    @property
    def board(self) -> np.ndarray:
        view = self._board.view()
        view.flags.writeable = False
        return view


class Game:
    """
    The Game class is the base class for any two-player, zero-sum, perfect
    information, abstract strategy game. Its derived classes need to implement
    game logic, symmetries, and action masks.

    Methods:
        start_state: returns the initial state of the game
        next_state: returns the next state of the game given a current state and action
        is_terminal: returns True if the game is over, False otherwise
        action_mask: returns a mask of legal actions for the player
        rewards: returns the rewards for the players
        num_symmetries: returns the number of symmetries for the game
        symmetries: generates game states symmetric to the given one
    """

    def start_state(self) -> GameState:
        """
        Returns the initial state of the game.
        """
        raise NotImplementedError

    def next_state(self, state: GameState, action: int) -> GameState:
        """
        Returns the next state of the game given a current state and action.
        Only valid if state is non-terminal and action is legal.
        """
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        raise NotImplementedError

    def action_mask(self, state: GameState) -> np.ndarray:
        """
        Returns a mask of legal actions for the player.
        """
        raise NotImplementedError
    
    def rewards(self, state: GameState) -> np.ndarray:
        """
        Returns a pair consisting of the rewards for the players.
        """
        raise NotImplementedError

    
    def num_symmetries(self) -> int:
        """
        Returns the number of symmetries for the game.
        """
        raise NotImplementedError

    def symmetries(self, state: GameState, symmetries: List[int]) -> List[GameState]:
        """
        Returns a list of symmetries for the state. The symmetries parameter
        is a list of integers representing the symmetries to apply to the state,
        and should be in the range [0, num_symmetries()).
        """
        raise NotImplementedError
