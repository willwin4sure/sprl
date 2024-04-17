import os

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agents.policy_agent import PolicyAgent
from src.agents.random_agent import RandomAgent
from src.evaluator.play import play
from src.games.connect_k import ConnectK
from src.interface.tracer import trace_model
from src.interface.run_self_play import run_self_play
from src.networks.new_connect_four_network import NewConnectFourNetwork
from src.policies.network_policy import NetworkPolicy

device = "cuda" if torch.cuda.is_available() else "cpu"

#######################
##  Hyperparameters  ##
#######################

RUN_NAME = "flamingo"
NUM_ITERS = 100
NUM_INIT_GAMES = 2500
NUM_GAMES_PER_ITER = 1000
NUM_PAST_ITERATIONS_TO_TRAIN = 10
NUM_EPOCHS = 150
BATCH_SIZE = 1024
UCT_INIT_ITERATIONS = 10000
UCT_ITERATIONS = 300  # total number of UCT iterations to run
MAX_TRAVERSALS = 16  # max traversals per batch
MAX_QUEUE_SIZE = 8  # max NN evals per batch

os.makedirs(f"data/games/{RUN_NAME}", exist_ok=True)
os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)

def train_network(network: NewConnectFourNetwork, iteration: int):
    """
    Train a network on the games generated from self-play.
    """
    all_states = []
    all_distributions = []
    all_outcomes = []
    
    for i in range(max(0, iteration - NUM_PAST_ITERATIONS_TO_TRAIN), iteration + 1):
        states = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_states.npy"))
        distributions = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_distributions.npy"))
        outcomes = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_outcomes.npy"))
        
        all_states.append(states)
        all_distributions.append(distributions)
        all_outcomes.append(outcomes)
    
    state_tensor = torch.cat(all_states, dim=0).to(device)
    distribution_tensor = torch.cat(all_distributions, dim=0).to(device)
    outcome_tensor = torch.cat(all_outcomes, dim=0).unsqueeze(1).to(device)
    
    dataset = TensorDataset(state_tensor, distribution_tensor, outcome_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    network.to(device)
    
    lr = 1e-3
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    
    with tqdm(range(NUM_EPOCHS)) as pbar:
        for epoch in pbar:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_state, batch_policy, batch_value in dataloader:
                policy_pred, value_pred = network(batch_state)

                policy_loss = torch.nn.functional.cross_entropy(
                    policy_pred, batch_policy)
                value_loss = torch.nn.functional.mse_loss(
                    value_pred, batch_value)

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

            pbar.set_description(
                f"Policy: {init_policy_loss:.6f} -> {average_policy_loss:.6f}, Value: {init_value_loss:.6f} -> {average_value_loss:.6f}")

    network.to("cpu")
    
    torch.save(
        network, f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt")
    
    trace_model(f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt",
                torch.randn(1, 2, 6, 7),
                f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt")
    

def train():
    game = ConnectK()
    network = NewConnectFourNetwork(2, 64)
    network_policy = NetworkPolicy(network, symmetrize=True)
    network_agent = PolicyAgent(network_policy, 0.1)
    random_agent = RandomAgent()
    
    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration}...")
        
        network_path = "random" if iteration == 0 \
            else f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration - 1}.pt"
            
        num_games = NUM_INIT_GAMES if iteration == 0 else NUM_GAMES_PER_ITER
        uct_iterations = UCT_INIT_ITERATIONS if iteration == 0 else UCT_ITERATIONS
        
        run_self_play(network_path,
                      f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}",
                      num_games, uct_iterations, MAX_TRAVERSALS, MAX_QUEUE_SIZE,
                      do_print_tqdm=True)
        
        train_network(network, iteration)
        
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
            

if __name__ == "__main__":
    train()
