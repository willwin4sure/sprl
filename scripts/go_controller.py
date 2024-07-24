"""
go_controller.py
"""

import json
import os
import sys
import tempfile
import time
from typing import List, Set, Tuple

import data_muncher
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from src.interface.tracer import trace_model
from src.networks.grid_networks import BasicGridNetwork

print("Alive.")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

NUM_ROWS = NUM_COLS = 6
ACTION_SIZE = NUM_ROWS * NUM_COLS + 1
HISTORY_SIZE = 8

with open(f"config/config_uct.json", "r") as f:
    config_uct = json.load(f)

with open("./config/config_selfplay.json", "r") as f:
    config_selfplay = json.load(f)
    MODEL_NAME = config_selfplay["modelName"]
    MODEL_VARIANT = config_selfplay["modelVariant"]
    NUM_GROUPS = config_selfplay["numGroups"]
    NUM_WORKER_TASKS = config_selfplay["numWorkerTasks"]
    NUM_ITERS = config_selfplay["numIters"]
    SYNC = config_selfplay["sync"]

with open("./config/config_controller.json", "r") as f:
    config_controller = json.load(f)
    WORLD_SIZE = config_controller["worldSize"]
    WORKER_TIME_TO_KILL = config_controller["workerTimeToKill"]
    MODEL_NUM_BLOCKS = config_controller["modelNumBlocks"]
    MODEL_NUM_CHANNELS = config_controller["modelNumChannels"]
    RESET_NETWORK = config_controller["resetNetwork"]
    LINEAR_WEIGHTING = config_controller["linearWeighting"]
    NUM_PAST_ITERS_TO_TRAIN = config_controller["numPastItersToTrain"]
    MAX_GROUPS = config_controller["maxGroups"]
    EPOCHS_PER_GROUP = config_controller["epochsPerGroup"]
    BATCH_SIZE = config_controller["batchSize"]
    LR_INIT = config_controller["lrInit"]
    LR_DECAY_FACTOR = config_controller["lrDecayFactor"]
    LR_MILESTONE_ITERS = config_controller["lrMilestoneIters"]


RUN_NAME = f"{MODEL_NAME}_{MODEL_VARIANT}"


def setup(rank: int, world_size: int):
    # This needs to be changed if we are using multiple machines.
    os.environ['MASTER_ADDR'] = 'localhost'
    # This can be any number.
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} world_size {world_size} initialized.")


def cleanup():
    dist.destroy_process_group()


epochify_time = 0
save_time = 0
trace_time = 0


def train_network(rank: int, world_size: int,
                  network: BasicGridNetwork, learning_rate: float, iteration: int,
                  state_tensor: torch.Tensor, distribution_tensor: torch.Tensor,
                  outcome_tensor: torch.Tensor, timestamp_tensor: torch.Tensor):
    global epochify_time, save_time, trace_time
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
                epochify_time -= time.time()
                train_average_policy_loss, train_average_value_loss = epochify(
                    network, train_dataloader, optimizer, train=True)

                val_average_policy_loss, val_average_value_loss = epochify(
                    network, val_dataloader, train=False)
                epochify_time += time.time()
                save_time -= time.time()

                # Use distributed all reduce to average the losses across all processes.
                torch.distributed.barrier()
                torch.distributed.all_reduce(train_average_policy_loss,
                                             op=dist.ReduceOp.SUM)
                torch.distributed.all_reduce(train_average_value_loss,
                                             op=dist.ReduceOp.SUM)
                torch.distributed.all_reduce(val_average_policy_loss,
                                             op=dist.ReduceOp.SUM)
                torch.distributed.all_reduce(val_average_value_loss,
                                             op=dist.ReduceOp.SUM)
                torch.distributed.barrier()

                train_average_policy_loss /= world_size
                train_average_value_loss /= world_size
                val_average_policy_loss /= world_size
                val_average_value_loss /= world_size

                val_loss = val_average_policy_loss + val_average_value_loss

                if val_loss < best_val_loss:
                    # If it is the best validation loss we've seen so far, save the model
                    best_val_loss = val_loss
                    best_epoch = epoch + group * EPOCHS_PER_GROUP
                    if rank == 0:
                        network.to("cpu")
                        torch.save(
                            network, f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt")
                        network.to(rank)

                pbar.set_description(
                    f"Tr Pol: {train_init_policy_loss:.4f} -> {train_average_policy_loss:.4f}, Tr Val: {train_init_value_loss:.4f} -> {train_average_value_loss:.4f}, Val Pol: {val_init_policy_loss:.4f} -> {val_average_policy_loss:.4f}, Val Val: {val_init_value_loss:.4f} -> {val_average_value_loss:.4f}")
                save_time += time.time()
            # If the best epoch is among the last EPOCHS_PER_GROUP // 2 epochs, don't break, might get more from training
            if best_epoch < (group + 1) * EPOCHS_PER_GROUP - EPOCHS_PER_GROUP // 2:
                break

    trace_time -= time.time()
    print(f"The best model was at epoch {best_epoch}.")

    trace_model(f"./data/models/{RUN_NAME}/{RUN_NAME}_iteration_{iteration}.pt",
                torch.randn(1, 2 * HISTORY_SIZE + 1, NUM_ROWS, NUM_COLS),
                f"./data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt")
    network.to("cpu")

    trace_time += time.time()

    print(
        f"Epochify: {epochify_time:.2f}s, Save: {save_time:.2f}s, Trace: {trace_time:.2f}s")


def epochify(rank: int, world_size: int,
             network: BasicGridNetwork, train_dataloader: DataLoader,
             optimizer: optim.Optimizer = None, train: bool = True, EPS: float = 1e-8) -> Tuple[float, float]:
    if train:
        network.train()
    else:
        network.eval()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # All-reduce the number of batches across all processes; only use the smallest number of batches
    num_batches = len(train_dataloader)
    torch.distributed.barrier()
    torch.distributed.all_reduce(num_batches, op=dist.ReduceOp.MIN)
    torch.distributed.barrier()

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


def main(rank, world_size):
    setup(rank, world_size)
    # Create the necessary directories
    os.makedirs(f"data/games/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"data/configs", exist_ok=True)

    print(f"Created necessary directories for {RUN_NAME}.")

    # Take that entire json file and write it to data/configs/...
    with open(f"data/configs/{RUN_NAME}_config_selfplay.json", "w") as f:
        json.dump(config_selfplay, f, indent=4)

    with open(f"data/configs/{RUN_NAME}_config_uct.json", "w") as f:
        json.dump(config_uct, f, indent=4)

    with open(f"data/configs/{RUN_NAME}_config_controller.json", "w") as f:
        json.dump(config_controller, f, indent=4)

    print(f"Saved configs for {RUN_NAME}.")

    network = BasicGridNetwork(
        NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS).to(rank)

    network = DDP(network, device_ids=[rank])
    print(f"Rank {rank} network is on device {network.device}")
    learning_rate = LR_INIT

    startup_time = time.time()
    # Remember this. From now on, startup_time is used for remembering when models finished training.

    timing_filepath = f"data/timings/{RUN_NAME}_timing.txt"
    timestamps = []

    muncher = data_muncher.DataMuncher(rank, world_size, NUM_WORKER_TASKS, NUM_GROUPS,
                                       RUN_NAME, LINEAR_WEIGHTING, WORKER_TIME_TO_KILL, SYNC, NUM_PAST_ITERS_TO_TRAIN)

    for iteration in range(NUM_ITERS):
        print(f"Starting iteration {iteration}...")

        if iteration in LR_MILESTONE_ITERS:
            print(f"LR {learning_rate} -> {learning_rate * LR_DECAY_FACTOR}")
            learning_rate *= LR_DECAY_FACTOR

        if RESET_NETWORK:
            print(f"Resetting network.")
            network = BasicGridNetwork(
                NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)

        train_dataset = muncher.get()
        train_network(rank, world_size,
                      network, learning_rate, iteration, **train_dataset)

        timestamps.append(time.time() - startup_time)
        if rank == 0:
            with open(timing_filepath, "w") as f:
                f.write("\n".join(str(t) for t in timestamps))

    torch.distributed.barrier()
    print(f"Rank {rank} finished training.")
    cleanup()


if __name__ == "__main__":
    mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    # main()
