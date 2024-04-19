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

NUM_ITERS = 200
NUM_GAMES_PER_ITER = 500
UCT_ITERATIONS = 512
MAX_TRAVERSALS = 8
MAX_QUEUE_SIZE = 4

INIT_NUM_GAMES = 2500
INIT_UCT_ITERATIONS = 32768
INIT_MAX_TRAVERSALS = 1
INIT_MAX_QUEUE_SIZE = 1

NUM_PAST_ITERS_TO_TRAIN = 20
MAX_GROUPS = 5
EPOCHS_PER_GROUP = 20
BATCH_SIZE = 1024


os.makedirs(f"data/games/{RUN_NAME}", exist_ok=True)
os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)


def train_network(network: NewConnectFourNetwork, iteration: int):
    """
    Train a network on the games generated from self-play.
    """
    network.to(device)

    all_states = []
    all_distributions = []
    all_outcomes = []
    all_timestamps = []
    
    timestamp = 1
    for i in range(max(0, iteration - NUM_PAST_ITERS_TO_TRAIN), iteration + 1):
        states = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_states.npy"))
        distributions = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_distributions.npy"))
        outcomes = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_outcomes.npy"))
        timestamps = torch.Tensor([timestamp for _ in range(states.shape[0])])
        
        timestamp += 1
        
        all_states.append(states)
        all_distributions.append(distributions)
        all_outcomes.append(outcomes)
        all_timestamps.append(timestamps)
    
    state_tensor = torch.cat(all_states, dim=0).to(device)
    distribution_tensor = torch.cat(all_distributions, dim=0).to(device)
    outcome_tensor = torch.cat(all_outcomes, dim=0).unsqueeze(1).to(device)
    timestamp_tensor = torch.cat(all_timestamps, dim=0).unsqueeze(1).to(device)
    
    # Split data into training and testing sets
    dataset = TensorDataset(state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)
    num_samples = len(dataset)
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders for training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    lr = 1e-3
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = 0

    for group in range(MAX_GROUPS):
        with tqdm(range(EPOCHS_PER_GROUP)) as pbar:
            for epoch in pbar:
                train_total_policy_loss = 0.0
                train_total_value_loss = 0.0
                train_num_batches = 0

                for batch_state, batch_policy, batch_value, batch_timestamp in train_dataloader:
                    policy_pred, value_pred = network(batch_state)
                    
                    # Softmax the policy prediction (network returns logits)
                    policy_pred = torch.softmax(policy_pred, dim=1)

                    # Weighted NLL loss by timestamp
                    policy_loss = torch.sum(-torch.sum(batch_policy * torch.log(policy_pred), dim=1, keepdim=True) * batch_timestamp) / torch.sum(batch_timestamp)

                    # Weighted MSE loss by timestamp
                    value_loss = torch.sum((batch_value - value_pred) ** 2 * batch_timestamp) / torch.sum(batch_timestamp)

                    loss = policy_loss + value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_total_policy_loss += policy_loss.item()
                    train_total_value_loss += value_loss.item()
                    train_num_batches += 1

                val_total_policy_loss = 0.0
                val_total_value_loss = 0.0
                val_num_batches = 0

                for batch_state, batch_policy, batch_value, batch_timestamp in val_dataloader:
                    policy_pred, value_pred = network(batch_state)
                    
                    # Softmax the policy prediction (network returns logits)
                    policy_pred = torch.softmax(policy_pred, dim=1)

                    # Weighted NLL loss by timestamp
                    policy_loss = torch.sum(-torch.sum(batch_policy * torch.log(policy_pred), dim=1, keepdim=True) * batch_timestamp) / torch.sum(batch_timestamp)

                    # Weighted MSE loss by timestamp
                    value_loss = torch.sum((batch_value - value_pred) ** 2 * batch_timestamp) / torch.sum(batch_timestamp)

                    val_total_policy_loss += policy_loss.item()
                    val_total_value_loss += value_loss.item()
                    val_num_batches += 1

                train_average_policy_loss = train_total_policy_loss / train_num_batches
                train_average_value_loss = train_total_value_loss / train_num_batches

                val_average_policy_loss = val_total_policy_loss / val_num_batches
                val_average_value_loss = val_total_value_loss / val_num_batches

                val_loss = val_average_policy_loss + val_average_value_loss
                
                if val_loss < best_val_loss:
                    # If it is the best validation loss we've seen so far, save the model
                    best_val_loss = val_loss
                    best_epoch = epoch + group * EPOCHS_PER_GROUP

                    network.to("cpu")
                    torch.save(network, f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt")
                    network.to(device)

                if epoch == 0:
                    train_init_policy_loss = train_average_policy_loss
                    train_init_value_loss = train_average_value_loss

                    val_init_policy_loss = val_average_policy_loss
                    val_init_value_loss = val_average_value_loss

                pbar.set_description(
                    f"Tr Pol: {train_init_policy_loss:.4f} -> {train_average_policy_loss:.4f}, Tr Val: {train_init_value_loss:.4f} -> {train_average_value_loss:.4f}, Val Pol: {val_init_policy_loss:.4f} -> {val_average_policy_loss:.4f}, Val Val: {val_init_value_loss:.4f} -> {val_average_value_loss:.4f}")
                
            # If the best epoch is among the last EPOCHS_PER_GROUP / 4 epochs, don't break, might get more from training
            if best_epoch < (group + 1) * EPOCHS_PER_GROUP - EPOCHS_PER_GROUP // 4:
                break

    print(f"The best model was at epoch {best_epoch}.")
    
    trace_model(f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt",
                torch.randn(1, 3, 6, 7),
                f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt")
    
    network.to("cpu")
    

def train():

    game = ConnectK()
    
    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration}...")

        network = NewConnectFourNetwork(2, 64)
        network_policy = NetworkPolicy(network, symmetrize=True)
        network_agent = PolicyAgent(network_policy, 0.0)
                
        network_path = "random" if iteration == 0 \
            else f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration - 1}.pt"
            
        num_games = INIT_NUM_GAMES if iteration == 0 else NUM_GAMES_PER_ITER
        uct_iterations = INIT_UCT_ITERATIONS if iteration == 0 else UCT_ITERATIONS
        max_traversals = INIT_MAX_TRAVERSALS if iteration == 0 else MAX_TRAVERSALS
        max_queue_size = INIT_MAX_QUEUE_SIZE if iteration == 0 else MAX_QUEUE_SIZE
        
        run_self_play(network_path,
                      f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}",
                      num_games, uct_iterations, max_traversals, max_queue_size,
                      do_print_tqdm=True)
        
        train_network(network, iteration)
        
        random_agent = RandomAgent()

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
