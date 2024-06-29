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
from src.networks.connect_four_network import ConnectFourNetwork
from src.policies.monte_carlo_policy import MonteCarloPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.random_policy import RandomPolicy
from src.policies.uct_policy import UCTPolicy

if __name__ == "__main__":
    connect4 = ConnectK()

    network1 = torch.load("data/models/flamingo/flamingo_iteration_9.pt")
    network_policy1 = NetworkPolicy(network1, symmetrize=True)
    uct_policy1 = UCTPolicy(
        network_policy1, num_iters=10000, c=1.0, train=False)  # , init_type="zero")
    policy_agent1 = PolicyAgent(uct_policy1, 0.5)

    # network2 = torch.load(
    #     "data/models/dragon/dragon_iteration_50.pt")
    # network_policy2 = NetworkPolicy(network2)
    # uct_policy2 = UCTPolicy(
    #     network_policy2, num_iters=100, c=1.0, train=False)
    # policy_agent2 = PolicyAgent(uct_policy2, 0.5)

    # policy1_wins = 0
    # policy2_wins = 0

    # with tqdm(total=50) as pbar:
    #     for _ in range(50):
    #         winner = play(connect4, (policy_agent1,
    #                       policy_agent2), do_print=False)

    #         if winner == 0:
    #             policy1_wins += 1
    #         else:
    #             policy2_wins += 1

    #         winner = play(connect4, (policy_agent2,
    #                       policy_agent1), do_print=False)

    #         if winner == 1:
    #             policy1_wins += 1
    #         else:
    #             policy2_wins += 1

    #         pbar.update(1)
    #         pbar.set_description(
    #             f"Policy 1 wins: {policy1_wins}, Policy 2 wins: {policy2_wins}")

    # monte_policy = MonteCarloPolicy(temperature=1.0, num_simulations=10)

    # uct_monte_policy = UCTPolicy(monte_policy, num_iters=1000, c=1.0, train=False)
    # policy_agent3 = PolicyAgent(uct_monte_policy, 0.1)

    agents = (
        # PolicyAgent(UCTPolicy(NetworkPolicy(torch.load("data/models/elephant/elephant_iteration_0.pt")), 1000), 0.1),
        PolicyAgent(uct_policy1, 0.1),
        HumanAgent()
    )

    play(connect4, agents, do_print=True)
