"""
uct_node.py

This module contains the UCTNode class, which represents a node in the UCT search tree.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Optional, Dict
import numpy as np

from src.games.game import Game, GameState


class UCTNode:

    def __init__(self, game: Game, game_state: GameState, action: int, num_actions: int, parent: 'UCTNode' = None):
        """
        Initialize a new UCTNode.
        """
        self.game: Game = game
        self.game_state: GameState = game_state
        self.action: int = action
        self.is_expanded: bool = False
        self.parent: Optional[UCTNode] = parent
        
        # This is a dictionary of action -> UCTNode. Only legal actions are keys.
        self.children: Dict[int, UCTNode] = {}

        # The priors and values are obtained from a neural network every time you expand a node
        # The priors, total values, and number visits will be 0 on all illegal actions
        self.child_priors: np.ndarray = np.zeros(num_actions)
        self.child_total_value: np.ndarray = np.zeros(num_actions)
        self.child_number_visits: np.ndarray = np.zeros(num_actions)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        """
        The value estimate for each child, based on the average value of all visits.
        """
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """
        The uncertainty for each child, based on the UCT formula (think UCB).
        """
        return np.sqrt(self.number_visits) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        """
        Compute the best legal child, with the action mask.
        """
        scores = self.child_Q() + self.child_U()

        # set all illegal actions to -inf
        scores[~self.game_state.action_mask()] = -np.inf

        return np.argmax(scores)
    
    def select_leaf(self) -> 'UCTNode':
        """
        Deterministically select the next leaf to expand based on the best path.
        """
        current = self

        # iterate until either you reach an un-expanded node or a terminal state
        while current.is_expanded and not self.game.is_terminal(current.game_state):
            current = current.best_child()

        return current
    
    def expand(self, child_priors):
        """
        Expand a non-terminal, un-expanded node using the child_priors from the neural network.
        """
        if self.game.is_terminal(self.game_state):
            raise ValueError("Cannot expand from a terminal state.")

        if self.is_expanded:
            raise ValueError("Cannot expand an already expanded node.")
        
        self.is_expanded = True

        action_mask = self.game.action_mask(self.game_state)
        for action, prior in enumerate(child_priors):
            if action_mask[action]:
                self.add_child(action, prior)

    def add_child(self, action, prior):
        """
        Add a child with a given action and prior probability.
        """
        if action in self.children:
            raise ValueError(f"Child with action {action} already exists.")
        
        self.child_priors[action] = prior
        
        self.children[action] = UCTNode(
            self.game.next_state(self.game_state, action),
            action,
            parent=self,
        )
    
    def backup(self, value_estimate):
        """
        Propagate the value_estimate of the current node, in a relative sense of the current player,
        back up along the path to the root.
        """
        # value estimate is negated here since values are from the perspective of parent
        absolute_estimate = -value_estimate * ((-1) ** self.game_state.player)
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += absolute_estimate * ((-1) ** current.game_state.player)
            
            current = current.parent

