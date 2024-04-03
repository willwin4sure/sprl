"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""

from typing import Tuple

import torch
from tqdm import tqdm

from src.agents.agent import Agent
from src.agents.human_agent import HumanAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.random_agent import RandomAgent

from src.evaluator.play import play

from src.games.connect_k import ConnectK
from src.games.game import Game, GameState

from src.networks.network import ConnectFourNetwork

from src.policies.monte_carlo_policy import MonteCarloPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.random_policy import RandomPolicy
from src.policies.uct_policy import UCTPolicy


if __name__ == "__main__":
    connect4 = ConnectK()
    
    network1 = torch.load("data/models/cheetah/cheetah_iteration_99.pt")
    network_policy1 = NetworkPolicy(network1)
    uct_policy1 = UCTPolicy(network_policy1, num_iters=1000, c=1.0, train=False)
    policy_agent1 = PolicyAgent(uct_policy1, 0.1)

    # network2 = torch.load("data/models/bison/bison_iteration_0.pt")
    # network_policy2 = NetworkPolicy(network2)
    # uct_policy2 = UCTPolicy(network_policy2, num_iters=100, c=1.0, train=False)
    # policy_agent2 = PolicyAgent(uct_policy2, 0.1)

    # monte_policy = MonteCarloPolicy(temperature=1.0, num_simulations=10)

    # uct_monte_policy = UCTPolicy(monte_policy, num_iters=1000, c=1.0, train=False)
    # policy_agent3 = PolicyAgent(uct_monte_policy, 0.1)

    agents = (
        policy_agent1,
        HumanAgent(),
    )    

    play(connect4, agents, do_print=True)
