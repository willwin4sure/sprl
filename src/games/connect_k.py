"""
connect_k.py

This module contains the ConnectK class, a derived class of the Game class
for a slight generalization of the classic game Connect Four.
"""

from typing import List
import numpy as np

from src.games.game import Game, GameState


ConnectKState = GameState  # type alias for immutable state for ConnectK game


class ConnectK(Game):
    """
    ConnectK class, a derived class of the Game class for a slight generalization
    of the classic game Connect Four.
    """

    def __init__(self, rows: int = 6, cols: int = 7, k: int = 4):
        """
        Initializes a Connect K game with a board of size rows x cols and
        a win condition of k in a row.

        Requires 1 <= rows, cols and 1 <= k <= min(rows, cols).
        """
        self.rows = rows
        self.cols = cols
        self.k = k
        self._win_idx = self._make_win_idx()  # cached win indices

    def start_state(self) -> ConnectKState:
        board = -np.ones(self.rows * self.cols)  # empty board
        return ConnectKState(board, 0, -1)  # player 0 to move

    def next_state(self, state: ConnectKState, action: int) -> ConnectKState:
        assert not self.is_terminal(state)
        assert self.action_mask(state)[action] == 1

        # make the move on the copy of the board
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
            return np.zeros(2)
        return np.array([1.0, -1.0]) if state.winner == 0 else np.array([-1.0, 1.0])

    def num_symmetries(self) -> int:
        # can only flip the board horizontally
        return 2

    def symmetries(self, state: ConnectKState, symmetries: List[int]) -> List[ConnectKState]:
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
    
    def display_state(self, state: ConnectKState) -> str:
        board2d = state.board.reshape(self.rows, self.cols)
        display = ""
        for row in range(self.rows):
            for col in range(self.cols):
                if board2d[row, col] == -1:
                    display += "."
                elif board2d[row, col] == 0:
                    display += "O"
                else:
                    display += "X"
                display += " "
            display += "\n"
        return display

    def _make_win_idx(self) -> np.ndarray:
        """
        Returns an array of indices for each possible winning line.
        """
        idx = []

        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - self.k + 1):
                base = row * self.cols + col
                idx.append(np.arange(base, base + self.k))

        # Vertical
        for row in range(self.rows - self.k + 1):
            for col in range(self.cols):
                base = row * self.cols + col
                idx.append(np.arange(base, base + self.k * self.cols, self.cols))

        # Diagonal (down-right)
        for row in range(self.rows - self.k + 1):
            for col in range(self.cols - self.k + 1):
                base = row * self.cols + col
                idx.append(
                    np.arange(base, base + self.k * (self.cols + 1), self.cols + 1)
                )

        # Diagonal (down-left)
        for row in range(self.rows - self.k + 1):
            for col in range(self.k - 1, self.cols):
                base = row * self.cols + col
                idx.append(
                    np.arange(base, base + self.k * (self.cols - 1), self.cols - 1)
                )

        return np.array(idx)
