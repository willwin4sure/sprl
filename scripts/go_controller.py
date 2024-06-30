"""
go_controller.py
"""

import json
import os
import time
from typing import Set

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from src.interface.tracer import trace_model
from src.networks.grid_networks import BasicGridNetwork

NUM_ROWS = 7
NUM_COLS = 7
ACTION_SIZE = 50
HISTORY_SIZE = 8

# Read all of the above from ./config/config_selfplay.json.
with open("./config/config_selfplay.json", "r") as f:
    config_selfplay = json.load(f)
    MODEL_NAME = config_selfplay["modelName"]
    MODEL_VARIANT = config_selfplay["modelVariant"]
    NUM_GROUPS = config_selfplay["numGroups"]
    NUM_WORKER_TASKS = config_selfplay["numWorkerTasks"]
    NUM_ITERS = config_selfplay["numIters"]

    controllerOptions = config_selfplay["controllerOptions"]
    WORKER_TIME_TO_KILL = controllerOptions["workerTimeToKill"]
    MODEL_NUM_BLOCKS = controllerOptions["modelNumBlocks"]
    MODEL_NUM_CHANNELS = controllerOptions["modelNumChannels"]
    RESET_NETWORK = controllerOptions["resetNetwork"]
    LINEAR_WEIGHTING = controllerOptions["linearWeighting"]
    NUM_PAST_ITERS_TO_TRAIN = controllerOptions["numPastItersToTrain"]
    MAX_GROUPS = controllerOptions["maxGroups"]
    EPOCHS_PER_GROUP = controllerOptions["epochsPerGroup"]
    BATCH_SIZE = controllerOptions["batchSize"]
    LR_INIT = controllerOptions["lrInit"]
    LR_DECAY_FACTOR = controllerOptions["lrDecayFactor"]
    LR_MILESTONE_ITERS = controllerOptions["lrMilestoneIters"]


RUN_NAME = f"{MODEL_NAME}_{MODEL_VARIANT}"


device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_data(iteration: int, live_workers: Set[int]):
    new_states = []
    new_distributions = []
    new_outcomes = []
    new_timestamps = []

    start_time = None

    finished_workers = set()

    while True:
        for task_id in live_workers:
            group = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)

            thread_save_path = f"data/games/{RUN_NAME}/{group}/{task_id}"

            # Check if the worker is unfinished and has saved its data.
            if (not task_id in finished_workers and
                os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_states.npy") and
                os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_distributions.npy") and
                    os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_outcomes.npy")):

                # Worker has finished collecting its data.
                finished_workers.add(task_id)

                states = torch.Tensor(
                    np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_states.npy"))
                distributions = torch.Tensor(np.load(
                    f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_distributions.npy"))
                outcomes = torch.Tensor(
                    np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}_outcomes.npy"))
                timestamps = torch.Tensor(
                    [iteration + 1 if LINEAR_WEIGHTING else 1 for _ in range(states.shape[0])])

                assert states.shape[0] == distributions.shape[0] == outcomes.shape[0] == timestamps.shape[0]

                new_states.append(states)
                new_distributions.append(distributions)
                new_outcomes.append(outcomes)
                new_timestamps.append(timestamps)

        if start_time is None and len(live_workers) > len(finished_workers) > len(live_workers) // 2:
            # Over half of the workers have finished, start a timer after which we will kill the rest
            print(
                f"Over half of the workers have finished. Starting timer to kill the rest.")
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

        print(
            f"Spinning on workers to finish... {len(finished_workers)} / {len(live_workers)} are complete.")
        time.sleep(30)

    print(
        f"Total samples for iteration {iteration}: {sum([s.shape[0] for s in new_states])}")

    return new_states, new_distributions, new_outcomes, new_timestamps


def train_network(network: BasicGridNetwork, learning_rate: float, iteration: int, state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor):
    network.to(device)

    dataset = TensorDataset(
        state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)

    # Split data into training and validation sets
    num_samples = len(dataset)
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and testing sets
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0

    train_init_policy_loss, train_init_value_loss = epochify(
        network, train_dataloader, train=False)

    val_init_policy_loss, val_init_value_loss = epochify(
        network, val_dataloader, train=False)

    for group in range(MAX_GROUPS):
        with tqdm(range(EPOCHS_PER_GROUP)) as pbar:
            for epoch in pbar:

                train_average_policy_loss, train_average_value_loss = epochify(
                    network, train_dataloader, optimizer, train=True)

                val_average_policy_loss, val_average_value_loss = epochify(
                    network, val_dataloader, train=False)

                val_loss = val_average_policy_loss + val_average_value_loss

                if val_loss < best_val_loss:
                    # If it is the best validation loss we've seen so far, save the model
                    best_val_loss = val_loss
                    best_epoch = epoch + group * EPOCHS_PER_GROUP

                    network.to("cpu")
                    torch.save(
                        network, f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt")
                    network.to(device)

                pbar.set_description(
                    f"Tr Pol: {train_init_policy_loss:.4f} -> {train_average_policy_loss:.4f}, Tr Val: {train_init_value_loss:.4f} -> {train_average_value_loss:.4f}, Val Pol: {val_init_policy_loss:.4f} -> {val_average_policy_loss:.4f}, Val Val: {val_init_value_loss:.4f} -> {val_average_value_loss:.4f}")

            # If the best epoch is among the last EPOCHS_PER_GROUP // 2 epochs, don't break, might get more from training
            if best_epoch < (group + 1) * EPOCHS_PER_GROUP - EPOCHS_PER_GROUP // 2:
                break

    print(f"The best model was at epoch {best_epoch}.")

    trace_model(f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt",
                torch.randn(1, 2 * HISTORY_SIZE + 1, NUM_ROWS, NUM_COLS),
                f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt")

    network.to("cpu")


def epochify(network: BasicGridNetwork, train_dataloader, optimizer=None, train=True, EPS=1e-8):
    if train:
        network.train()
    else:
        network.eval()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for batch_state, batch_policy, batch_value, batch_timestamp in train_dataloader:
        policy_pred, value_pred = network(batch_state)

        # Softmax the policy prediction (network returns logits)
        policy_pred = torch.softmax(policy_pred, dim=1)

        # Weighted NLL loss by timestamp
        policy_loss = torch.sum(-torch.sum(batch_policy * torch.log(
            policy_pred + EPS), dim=1, keepdim=True) * batch_timestamp) / torch.sum(batch_timestamp)

        # Weighted MSE loss by timestamp
        value_loss = torch.sum(
            (batch_value - value_pred) ** 2 * batch_timestamp) / torch.sum(batch_timestamp)

        if train:
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    average_policy_loss = total_policy_loss / num_batches
    average_value_loss = total_value_loss / num_batches

    return average_policy_loss, average_value_loss


def main():
    print(f"I have access to {device}.")

    # Create the necessary directories
    os.makedirs(f"data/games/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/configs", exist_ok=True)

    print(f"Created necessary directories for {RUN_NAME}.")

    # Take that entire json file and write it to data/configs/...
    with open(f"data/configs/{RUN_NAME}_config_selfplay.json", "w") as f:
        json.dump(config_selfplay, f, indent=4)

    print(
        f"Wrote config file to data/configs/{RUN_NAME}_config_selfplay.json."
    )

    # read config from config_uct
    with open(f"config/config_uct.json", "r") as f:
        config_uct = json.load(f)
    # same thing, with uct
    with open(f"data/configs/{RUN_NAME}_config_uct.json", "w") as f:
        json.dump(config_uct, f, indent=4)

    all_state_tensors = []
    all_distribution_tensors = []
    all_outcome_tensors = []
    all_timestamp_tensors = []

    network = BasicGridNetwork(
        NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)
    learning_rate = LR_INIT

    live_workers = set(range(NUM_WORKER_TASKS))

    for iteration in range(NUM_ITERS):
        print(f"Starting iteration {iteration}...")

        if iteration in LR_MILESTONE_ITERS:
            print(f"Decaying learning rate to {learning_rate}.")
            learning_rate *= LR_DECAY_FACTOR

        if RESET_NETWORK:
            print(f"Resetting network.")
            network = BasicGridNetwork(
                NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)

        new_states, new_distributions, new_outcomes, new_timestamps = collate_data(
            iteration, live_workers)

        new_state_tensor = torch.cat(new_states, dim=0).to(device)
        new_distribution_tensor = torch.cat(
            new_distributions, dim=0).to(device)
        new_outcome_tensor = torch.cat(
            new_outcomes, dim=0).unsqueeze(1).to(device)
        new_timestamp_tensor = torch.cat(
            new_timestamps, dim=0).unsqueeze(1).to(device)

        assert new_state_tensor.shape[0] == new_distribution_tensor.shape[
            0] == new_outcome_tensor.shape[0] == new_timestamp_tensor.shape[0]

        all_state_tensors.append(new_state_tensor)
        all_distribution_tensors.append(new_distribution_tensor)
        all_outcome_tensors.append(new_outcome_tensor)
        all_timestamp_tensors.append(new_timestamp_tensor)

        assert (len(all_state_tensors) == len(all_distribution_tensors)
                == len(all_outcome_tensors) == len(all_timestamp_tensors))

        while len(all_state_tensors) > NUM_PAST_ITERS_TO_TRAIN:
            all_state_tensors.pop(0)
            all_distribution_tensors.pop(0)
            all_outcome_tensors.pop(0)
            all_timestamp_tensors.pop(0)

        train_state_tensor = torch.cat(all_state_tensors, dim=0)
        train_distribution_tensor = torch.cat(all_distribution_tensors, dim=0)
        train_outcome_tensor = torch.cat(all_outcome_tensors, dim=0)
        train_timestamp_tensor = torch.cat(
            all_timestamp_tensors, dim=0) - max(0, iteration + 1 - NUM_PAST_ITERS_TO_TRAIN)

        assert train_state_tensor.shape[0] == train_distribution_tensor.shape[
            0] == train_outcome_tensor.shape[0] == train_timestamp_tensor.shape[0]
        assert torch.min(train_timestamp_tensor) > 0

        train_network(network, learning_rate, iteration, train_state_tensor,
                      train_distribution_tensor, train_outcome_tensor, train_timestamp_tensor)


if __name__ == "__main__":
    main()
