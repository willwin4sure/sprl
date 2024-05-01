"""
pentago_controller.py
"""

import os
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.interface.tracer import trace_model
from src.networks.pentago_network import PentagoNetwork

from src.pentago_constants import (
    NUM_ITERS,

    NUM_WORKER_TASKS,
    NUM_GROUPS,
    WORKER_TIME_TO_KILL,

    NUM_GAMES_PER_WORKER,
    UCT_ITERATIONS,
    MAX_TRAVERSALS,
    MAX_QUEUE_SIZE,

    INIT_NUM_GAMES_PER_WORKER,
    INIT_UCT_ITERATIONS,
    INIT_MAX_TRAVERSALS,
    INIT_MAX_QUEUE_SIZE,

    MODEL_NUM_BLOCKS,
    MODEL_NUM_CHANNELS,
    RESET_NETWORK,
    LINEAR_WEIGHTING,
    NUM_PAST_ITERS_TO_TRAIN,
    MAX_GROUPS,
    EPOCHS_PER_GROUP,
    BATCH_SIZE,
    LR,

    RUN_NAME,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def collate_data(iteration: int, live_workers: set):
    new_states = []
    new_distributions = []
    new_outcomes = []
    new_timestamps = []

    start_time = None

    while True:
        finished_workers = set()

        for task_id in live_workers:
            group = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)

            thread_save_path = f"data/games/{RUN_NAME}/{group}/{task_id}"

            if (os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_states.npy") and
                os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_distributions.npy") and
                os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_outcomes.npy")):
                
                # Worker has finished collecting its data
                finished_workers.add(task_id)

                states = torch.Tensor(np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_states.npy"))
                distributions = torch.Tensor(np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_distributions.npy"))
                outcomes = torch.Tensor(np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_outcomes.npy"))
                timestamps = torch.Tensor([iteration + 1 if LINEAR_WEIGHTING else 1 for _ in range(states.shape[0])])

                assert states.shape[0] == distributions.shape[0] == outcomes.shape[0] == timestamps.shape[0]

                new_states.append(states)
                new_distributions.append(distributions)
                new_outcomes.append(outcomes)
                new_timestamps.append(timestamps)

        if start_time is None and len(live_workers) > len(finished_workers) > len(live_workers) // 2:
            # Over half of the workers have finished, start a timer after which we will kill the rest
            print(f"Over half of the workers have finished. Starting timer to kill the rest.")
            start_time = time.time()

        if start_time is not None and time.time() - start_time > WORKER_TIME_TO_KILL:
            # Kill the remaining workers after time has expired
            for task_id in live_workers - finished_workers:
                live_workers.remove(task_id)
            
            print(f"Killing unfinished workers: {len(live_workers)} left.")
            break

        if len(finished_workers) == len(live_workers):
            # All workers have finished
            break

        print(f"Spinning on workers to finish... {len(finished_workers)} / {len(live_workers)} are complete.")
        time.sleep(5)

    print(f"Total samples for iteration {iteration}: {sum([s.shape[0] for s in new_states])}")

    return new_states, new_distributions, new_outcomes, new_timestamps


def train_network(network: PentagoNetwork, iteration: int, state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor):
    network.to(device)

    dataset = TensorDataset(state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)

    # Split data into training and validation sets
    num_samples = len(dataset)
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders for training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR)

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
                
            # If the best epoch is among the last EPOCHS_PER_GROUP // 4 epochs, don't break, might get more from training
            if best_epoch < (group + 1) * EPOCHS_PER_GROUP - EPOCHS_PER_GROUP // 4:
                break

    print(f"The best model was at epoch {best_epoch}.")
    
    trace_model(f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt",
                torch.randn(1, 3, 6, 6),
                f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt")
    
    network.to("cpu")

def main():
    print(f"I am the controller. I have access to {device}.")

    # Create the necessary directories
    os.makedirs(f"data/games/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/configs", exist_ok=True)

    # Write the config of the run to a file
    with open(f"./data/configs/{RUN_NAME}_config.txt", "w") as f:
        f.write(f"NUM_ITERS = {NUM_ITERS}\n")
        f.write(f"NUM_GROUPS = {NUM_GROUPS}\n")
        f.write(f"NUM_WORKER_TASKS = {NUM_WORKER_TASKS}\n")
        f.write(f"WORKER_TIME_TO_KILL = {WORKER_TIME_TO_KILL}\n")
        f.write(f"NUM_GAMES_PER_WORKER = {NUM_GAMES_PER_WORKER}\n")
        f.write(f"UCT_ITERATIONS = {UCT_ITERATIONS}\n")
        f.write(f"MAX_TRAVERSALS = {MAX_TRAVERSALS}\n")
        f.write(f"MAX_QUEUE_SIZE = {MAX_QUEUE_SIZE}\n")
        f.write(f"INIT_NUM_GAMES_PER_WORKER = {INIT_NUM_GAMES_PER_WORKER}\n")
        f.write(f"INIT_UCT_ITERATIONS = {INIT_UCT_ITERATIONS}\n")
        f.write(f"INIT_MAX_TRAVERSALS = {INIT_MAX_TRAVERSALS}\n")
        f.write(f"INIT_MAX_QUEUE_SIZE = {INIT_MAX_QUEUE_SIZE}\n")
        f.write(f"MODEL_NUM_BLOCKS = {MODEL_NUM_BLOCKS}\n")
        f.write(f"MODEL_NUM_CHANNELS = {MODEL_NUM_CHANNELS}\n")
        f.write(f"RESET_NETWORK = {RESET_NETWORK}\n")
        f.write(f"LINEAR_WEIGHTING = {LINEAR_WEIGHTING}\n")
        f.write(f"NUM_PAST_ITERS_TO_TRAIN = {NUM_PAST_ITERS_TO_TRAIN}\n")
        f.write(f"MAX_GROUPS = {MAX_GROUPS}\n")
        f.write(f"EPOCHS_PER_GROUP = {EPOCHS_PER_GROUP}\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"LR = {LR}\n")

    all_states = []
    all_distributions = []
    all_outcomes = []
    all_timestamps = []

    network = PentagoNetwork(MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)

    live_workers = set(range(NUM_WORKER_TASKS))

    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration}...")

        if RESET_NETWORK:
            network = PentagoNetwork(MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)

        new_states, new_distributions, new_outcomes, new_timestamps = collate_data(iteration, live_workers)

        all_states.extend(new_states)
        all_distributions.extend(new_distributions)
        all_outcomes.extend(new_outcomes)
        all_timestamps.extend(new_timestamps)

        state_tensor = torch.cat(all_states, dim=0).to(device)
        distribution_tensor = torch.cat(all_distributions, dim=0).to(device)
        outcome_tensor = torch.cat(all_outcomes, dim=0).unsqueeze(1).to(device)
        timestamp_tensor = torch.cat(all_timestamps, dim=0).unsqueeze(1).to(device)

        assert state_tensor.shape[0] == distribution_tensor.shape[0] == outcome_tensor.shape[0] == timestamp_tensor.shape[0]

        train_network(network, iteration, state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)


if __name__ == "__main__":
    main()
