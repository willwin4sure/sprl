"""
game.py

This module contains the Game class, the abstract base class for any two-player,
zero-sum, perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class GameState:
    """
    Immutable base class to store the state of a game.
    """
    _board: np.ndarray = None  # private field, do not access directly
    player: int = 0
    winner: int = -1

    @property
    def board(self) -> np.ndarray:
        """
        Read-only property for the game board.
        """
        view = self._board.view()
        view.flags.writeable = False
        return view


class Game(ABC):
    """
    The Game class is the abstract base class for any two-player, zero-sum,
    perfect information, abstract strategy game.
    
    Derived classes need to implement game logic, symmetries, and action masks.

    Methods:
        start_state: returns initial state of the game
        next_state: returns next state of the game given current state and action
        is_terminal: returns True if the game is over, False otherwise
        action_mask: returns a mask of legal actions for the player
        rewards: returns the rewards for the players
        num_symmetries: returns the number of symmetries for the game
        symmetries: generates game states symmetric to the given one
        display_state: returns a string representation of the game state
        hash_state: returns a hash of the game state
    """

    @abstractmethod
    def start_state(self) -> GameState:
        """
        Returns the initial state of the game.
        """
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: GameState, action: int) -> GameState:
        """
        Returns the next state of the game given a current state and action.
        
        Requires that the state is non-terminal and action is legal.
        If the next state is terminal, the board, player, and winner still must
        be updated. The player should be the opposite of the player that just moved.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def action_mask(self, state: GameState) -> np.ndarray:
        """
        Returns a mask of legal actions for the player.
        
        Requires that the state is non-terminal.
        """
        raise NotImplementedError

    @abstractmethod
    def rewards(self, state: GameState) -> np.ndarray:
        """
        Returns a pair consisting of the rewards for the players.
        """
        raise NotImplementedError

    @abstractmethod
    def num_symmetries(self) -> int:
        """
        Returns the number of symmetries for the game.
        """
        raise NotImplementedError

    @abstractmethod
    def symmetries(self, state: GameState, symmetries: List[int]) -> List[GameState]:
        """
        Returns a list of states symmetric to the given state.
        
        The symmetries parameter is a list of integers representing
        the symmetries to apply to the state, and should be
        in the range [0, num_symmetries()).
        """
        raise NotImplementedError

    @abstractmethod
    def display_state(self, state: GameState) -> str:
        """
        Returns a string representation of the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def hash_state(self, state: GameState) -> int:
        """
        Returns a hash of the game state.
        """
        raise NotImplementedError
