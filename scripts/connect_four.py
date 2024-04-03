"""
connect_four.py

Putting it all together to train a Connect Four bot.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.games.game import Game, GameState
from src.games.connect_k import ConnectK

from src.networks.connect_four_network import ConnectFourNetwork

from src.policies.random_policy import RandomPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.uct_policy import UCTPolicy

from src.train.self_play import run_iteration

from src.agents.policy_agent import PolicyAgent
from src.agents.random_agent import RandomAgent

from src.evaluator.play import play

device = "cuda" if torch.cuda.is_available() else "cpu"

#######################
##  Hyperparameters  ##
#######################

# RUN_NAME = "dragon_mini"
# NUM_ITERS = 20
# NUM_INIT_GAMES = 200
# NUM_GAMES_PER_ITER = 50
# NUM_PAST_ITERATIONS_TO_TRAIN = 10
# NUM_EPOCHS = 10
# BATCH_SIZE = 1024
# UCT_TRAVERSALS = 10
# EXPLORATION = 2.0

RUN_NAME = "dragon"
NUM_ITERS = 100
NUM_INIT_GAMES = 2500
NUM_GAMES_PER_ITER = 500
NUM_PAST_ITERATIONS_TO_TRAIN = 20
NUM_EPOCHS = 300
BATCH_SIZE = 1024
UCT_TRAVERSALS = 200
EXPLORATION = 2.0

def train_network(game: Game, network: ConnectFourNetwork, iteration: int):
    """
    Train a network on the games generated from self-play.
    """
    states = []
    distributions = []
    rewards = []

    for i in range(max(0, iteration - NUM_PAST_ITERATIONS_TO_TRAIN), iteration + 1):
        states_i, distributions_i, rewards_i = torch.load(f"data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}.pkl")
        states.extend(states_i)
        distributions.extend(distributions_i)
        rewards.extend(rewards_i)

    state_tensors = []
    policy_tensors = []
    value_tensors = []

    for i in range(0, len(states)):
        state: GameState = states[i]
        distribution: np.ndarray = distributions[i]
        reward: float = rewards[i]

        state_tensors.append(network.embed(game, state).squeeze(0))
        policy_tensors.append(torch.tensor(distribution, dtype=torch.float32))
        value_tensors.append(torch.tensor([reward], dtype=torch.float32))

    state_tensor = torch.stack(state_tensors).to(device)
    policy_tensor = torch.stack(policy_tensors).to(device)
    value_tensor = torch.stack(value_tensors).to(device)

    dataset = TensorDataset(state_tensor, policy_tensor, value_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    network.to(device)

    lr = 1e-2
    if iteration >= 10:
        lr = 1e-3
    if iteration >= 25:
        lr = 1e-4

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    with tqdm(range(NUM_EPOCHS)) as pbar:
        for epoch in pbar:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_state, batch_policy, batch_value in dataloader:
                policy_pred, value_pred = network(batch_state)

                policy_loss = torch.nn.functional.cross_entropy(policy_pred, batch_policy)
                value_loss = torch.nn.functional.mse_loss(value_pred, batch_value)

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

            average_policy_loss = total_policy_loss / num_batches
            average_value_loss = total_value_loss / num_batches

            if epoch == 0:
                init_policy_loss = average_policy_loss
                init_value_loss = average_value_loss

            pbar.set_description(f"Policy: {init_policy_loss:.6f} -> {average_policy_loss:.6f}, Value: {init_value_loss:.6f} -> {average_value_loss:.6f}")

    network.to("cpu")

    torch.save(network, f"data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt")


def train():
    """
    Train a Connect Four bot.
    """

    game = ConnectK()
    network = ConnectFourNetwork()

    random_policy = RandomPolicy()
    network_policy = NetworkPolicy(network)
    uct_policy = UCTPolicy(network_policy, UCT_TRAVERSALS, c=EXPLORATION)
    uct_random_policy = UCTPolicy(random_policy, UCT_TRAVERSALS, c=EXPLORATION)

    # uct_win_counts = []
    network_win_counts = []

    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration}...")

        if iteration == 0:
            policy_to_use = uct_random_policy
            num_games_to_use = NUM_INIT_GAMES
        else:
            policy_to_use = uct_policy
            num_games_to_use = NUM_GAMES_PER_ITER

        states, distributions, rewards = run_iteration(game, (policy_to_use, policy_to_use), num_games_to_use)

        torch.save((states, distributions, rewards), f"data/games/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pkl")

        train_network(game, network, iteration)

        random_agent = RandomAgent()

        network_agent = PolicyAgent(network_policy, 0)

        network_wins = 0

        with tqdm(total=100) as pbar:
            for _ in range(50):
                winner = play(game, (network_agent, random_agent))
                if winner == 0:
                    network_wins += 1
                pbar.update(1)
                pbar.set_description(f"Network wins: {network_wins}")

            for _ in range(50):
                winner = play(game, (random_agent, network_agent))
                if winner == 1:
                    network_wins += 1
                pbar.update(1)
                pbar.set_description(f"Network wins: {network_wins}")

        # uct_win_counts.append(uct_wins)
        network_win_counts.append(network_wins)

    # save the win counts
    torch.save((network_win_counts), f"data/{RUN_NAME}_win_counts.pkl")

    # plot the win counts
    # plt.plot(uct_win_counts, label="UCT")
    plt.plot(network_win_counts, label="Network")
    plt.xlabel("Iteration")
    plt.ylabel("Win count")
    plt.legend()
    plt.savefig(f"data/{RUN_NAME}_win_counts.png")






        
        

if __name__ == "__main__":
    train()




