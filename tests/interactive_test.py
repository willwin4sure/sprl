"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""

from typing import Tuple
import torch
import time

from src.games.connect_k import ConnectK
from src.games.game import Game, GameState

from src.networks.network import ConnectFourNetwork

from src.policies.random_policy import RandomPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.uct_policy import UCTPolicy

from src.agents.agent import Agent
from src.agents.human_agent import HumanAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.random_agent import RandomAgent

# TODO: move this to evaluator, with options
# such as whether to print each state, logging, etc.
# then this test can just call that play function
# with the appropriate agents
def play(game: Game, agents: Tuple[Agent, Agent]):
    """
    Play an interactive game between two agents.
    """
    state: GameState = game.start_state()

    # main game loop
    while not game.is_terminal(state):
        print(f"Current state:\n{game.display_state(state)}\n")
        print(f"Player {state.player}'s turn")

        action_mask = game.action_mask(state)
        print(f"Legal action mask: {action_mask}")

        action = agents[state.player].action(game, state)

        state = game.next_state(state, action)

    print(f"Final state:\n{game.display_state(state)}\n\n")
    if state.winner == -1:
        print("Game ended in a draw.")
    else:
        print(f"Player {state.winner} wins!")

    print(f"Rewards: {game.rewards(state)}")


if __name__ == "__main__":
    connect4 = ConnectK()

    network1 = torch.load("data/models/alpaca/alpaca_iteration_49.pt")
    network_policy1 = NetworkPolicy(network1)
    uct_policy1 = UCTPolicy(network_policy1, num_iters=1000, c=1.0)
    policy_agent1 = PolicyAgent(uct_policy1, 0.1)

    # network2 = torch.load("data/models/alpaca/alpaca_iteration_0.pt")
    # network_policy2 = NetworkPolicy(network2)
    # uct_policy2 = UCTPolicy(network_policy2, num_iters=1000, c=1.0)
    # policy_agent2 = PolicyAgent(uct_policy2)

    agents = (
        policy_agent1,
        HumanAgent(),
        # policy_agent2,
    )    

    play(connect4, agents)

    

