"""
This script is used to reconstruct connect four states.
"""


import numpy as np
import torch

from src.games.connect_k import ConnectKState, ConnectK

from src.policies.network_policy import NetworkPolicy
from src.policies.uct_policy import UCTPolicy

def reconstruct_state(input: str) -> np.ndarray:
    input = ''.join(input.split())
    mapping = {'.': -1, 'O': 0, 'X': 1}
    return np.array([mapping[char] for char in input])

s = """
. . . . . . .
. . . . . . .
. . . O O . .
. . . X X O X
. . . X O X O
. . X O O O X
"""

connect4 = ConnectK()

network1 = torch.load("data/models/cheetah/cheetah_iteration_99.pt")
network_policy1 = NetworkPolicy(network1)
uct_policy1 = UCTPolicy(network_policy1, num_iters=100, c=1.0, train=False)

state = ConnectKState(reconstruct_state(s), 1)

print(network_policy1.action(connect4, state))
print(uct_policy1.action(connect4, state))