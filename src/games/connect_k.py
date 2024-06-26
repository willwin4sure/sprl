"""
connect_k.py

This module contains the ConnectK class, a derived class of the Game class
implementing a slight generalization of the classic game Connect Four.
"""

from typing import List

import numpy as np

from src.games.game import Game, GameState

ConnectKState = GameState  # type alias for immutable state for ConnectK game


class ConnectK(Game):
    """
    Derived class of the Game class implementing a slight
    generalization of the classic game Connect Four.
    """

    def __init__(self, rows: int = 6, cols: int = 7, k: int = 4):
        """
        Initializes a Connect K game with a board of size rows x cols and
        a win condition of k in a row.

        Requires 1 <= rows, cols and 1 <= k <= min(rows, cols).
        """
        assert rows >= 1 and cols >= 1 and 1 <= k <= min(rows, cols)
        
        self.rows = rows
        self.cols = cols
        self.k = k
        self._win_idx = self._make_win_idx()  # cached win indices

    def start_state(self) -> ConnectKState:
        board = -np.ones(self.rows * self.cols)  # empty board
        return ConnectKState(board, 0, -1)  # player 0 to move, no winner

    def next_state(self, state: ConnectKState, action: int) -> ConnectKState:
        assert not self.is_terminal(state)
        assert self.action_mask(state)[action] == 1

        # make the move on a copy of the board
        board2d = state.board.copy().reshape(self.rows, self.cols)
        num_filled = (board2d[:, action] != -1).sum()
        board2d[self.rows - num_filled - 1, action] = state.player
        board = board2d.flatten()

        # check if the move wins the game
        win = (board[self._win_idx] == state.player).all(axis=1).any()
        winner = state.player if win else -1

        return ConnectKState(board, 1 - state.player, winner)

    def is_terminal(self, state: ConnectKState) -> bool:
        return state.winner != -1 or (state.board != -1).all()

    def action_mask(self, state: ConnectKState) -> np.ndarray:
        board2d = state.board.reshape(self.rows, self.cols)
        return board2d[0] == -1

    def rewards(self, state: ConnectKState) -> np.ndarray:
        if state.winner == -1:
            # neither player has won yet
            return np.zeros(2)
        
        return np.array([1.0, -1.0]) if state.winner == 0 else np.array([-1.0, 1.0])

    def num_symmetries(self) -> int:
        # can only flip the board horizontally
        return 2
    
    def inverse_symmetry(self, symmetry: int) -> int:
        return symmetry

    def symmetrize_state(self, state: ConnectKState, symmetries: List[int]) -> List[ConnectKState]:
        # there is only one symmetry, encoded by the integer 1
        board2d = state.board.reshape(self.rows, self.cols)
        sym_states = []
        
        for sym in symmetries:
            if sym == 0:
                sym_states.append(state)
            if sym == 1:
                sym_states.append(
                    ConnectKState(np.flip(board2d, axis=1).flatten(), state.player, state.winner))
        
        return sym_states
    
    def symmetrize_action_distribution(self, action_distribution: np.ndarray, symmetries: List[int]) -> List[np.ndarray]:
        # there is only one symmetry, encoded by the integer 1
        sym_action_distributions = []
        
        for sym in symmetries:
            if sym == 0:
                sym_action_distributions.append(action_distribution.copy())
            if sym == 1:
                sym_action_distributions.append(np.flip(action_distribution).copy())
        
        return sym_action_distributions

    def display_state(self, state: ConnectKState) -> str:
        board2d = state.board.reshape(self.rows, self.cols)
        display = ""
        for row in range(self.rows):
            for col in range(self.cols):
                if board2d[row, col] == -1:
                    display += "."
                elif board2d[row, col] == 0:
                    # colored red
                    display += "\033[91mO\033[0m"
                else:
                    # colored yellow
                    display += "\033[93mX\033[0m"
                display += " "
            display += "\n"

        for col in range(self.cols):
            display += str(col) + " "

        return display


    def _make_win_idx(self) -> np.ndarray:
        """
        Returns an array of indices for each possible winning line.
        """
        idx = []

        # horizontal
        for row in range(self.rows):
            for col in range(self.cols - self.k + 1):
                base = row * self.cols + col
                idx.append(np.arange(base, base + self.k))

        # vertical
        for row in range(self.rows - self.k + 1):
            for col in range(self.cols):
                base = row * self.cols + col
                idx.append(
                    np.arange(base, base + self.k * self.cols, self.cols))

        # diagonal (down-right)
        for row in range(self.rows - self.k + 1):
            for col in range(self.cols - self.k + 1):
                base = row * self.cols + col
                idx.append(
                    np.arange(base, base + self.k *
                              (self.cols + 1), self.cols + 1)
                )

        # diagonal (down-left)
        for row in range(self.rows - self.k + 1):
            for col in range(self.k - 1, self.cols):
                base = row * self.cols + col
                idx.append(
                    np.arange(base, base + self.k *
                              (self.cols - 1), self.cols - 1)
                )

        return np.array(idx)

    def hash_state(self, state: GameState) -> int:
        return 0  # hash is not needed for this game
