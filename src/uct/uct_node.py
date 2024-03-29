"""
uct_node.py

This module contains the UCTNode class, which represents a node in the UCT search tree.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Optional, Dict
import numpy as np

from src.games.game import Game, GameState


class UCTNode:

    def __init__(self, game: Game, game_state: GameState, action: int, parent: 'UCTNode' = None):
        """
        Initialize a new UCTNode.
        """
        self.game: Game = game
        self.game_state: GameState = game_state
        self.action: int = action
        self.is_expanded: bool = False

        # Parent of the node, None if root
        self.parent: Optional[UCTNode] = parent
        
        # This is a dictionary of action -> UCTNode. Only legal actions are keys.
        self.children: Dict[int, UCTNode] = {}

        # Cached values so that we don't need to recompute them every time.
        self.action_mask = self.game.action_mask(self.game_state)
        self.is_terminal = self.game.is_terminal(self.game_state)

        # The priors and values are obtained from a neural network every time you expand a node
        # The priors, total values, and number visits will be 0 on all illegal actions
        num_actions = len(self.action_mask)
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
        return np.sqrt(1 + np.sum(self.child_number_visits)) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self, c: float = 1.0):
        """
        Compute the best legal child, with the action mask.
        """
        scores = self.child_Q() + c * self.child_U()

        # set all illegal actions to -inf
        scores[~self.action_mask] = -np.inf

        return self.children[np.argmax(scores)]
    
    def select_leaf(self, c: float = 1.0) -> 'UCTNode':
        """
        Deterministically select the next leaf to expand based on the best path.
        """
        current = self

        # iterate until either you reach an un-expanded node or a terminal state
        while current.is_expanded and not current.is_terminal:
            current = current.best_child(c)

        return current
    
    def expand(self, child_priors, train=True):
        """
        Expand a non-terminal, un-expanded node using the child_priors from the neural network.
        """
        # if self.game.is_terminal(self.game_state):
        #     raise ValueError("Cannot expand from a terminal state.")

        # if self.is_expanded:
        #     raise ValueError("Cannot expand an already expanded node.")
        
        self.is_expanded = True

        if train and self.action == -1:
            # if you are the root, add dirichlet noise to the prior
            places = self.action_mask > 0
            noise = np.random.dirichlet(0.03 * np.ones(np.sum(places)))

            noise_distribution = np.zeros_like(self.action_mask, dtype=np.float64)
            noise_distribution[places] = noise

            child_priors = 0.75 * child_priors + 0.25 * noise_distribution

        for action, prior in enumerate(child_priors):
            if self.action_mask[action]:
                self.add_child(action, prior)

    def add_child(self, action, prior):
        """
        Add a child with a given action and prior probability.
        """
        # if action in self.children:
        #     raise ValueError(f"Child with action {action} already exists.")
        
        self.child_priors[action] = prior
        
        self.children[action] = UCTNode(
            self.game,
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
        estimate = -value_estimate * ((-1) ** self.game_state.player)
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += estimate * ((-1) ** current.game_state.player)
            
            current = current.parent

