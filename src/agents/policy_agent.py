"""
policy_agent.py

An agent that samples from a policy to select actions,
using a temperature parameter.
"""

import numpy as np

from src.games.game import Game, GameState
from src.policies.policy import Policy
from src.agents.agent import Agent


class PolicyAgent(Agent):
    def __init__(self, policy: Policy):
        self.policy = policy

    def action(self, game: Game, state: GameState, temperature: float = 1.0) -> int:
        # note that action_probs only has support over valid actions
        action_probs, _ = self.policy.action(game, state)
        
        if temperature == 0:
            # deterministically pick best action
            action = np.argmax(action_probs)

        else:
            # reweight actions by raising probs to 1 / T and normalizing
            action_probs = np.power(action_probs, 1.0 / temperature)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(len(action_probs), p=action_probs)
        
        return action
