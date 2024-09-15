"""
go_controller.py
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from socket import gethostname
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

NUM_ROWS = NUM_COLS = 9
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
    BOOTSTRAP_NAME = config_controller["bootstrap"]

    WORLD_SIZE = config_controller["worldSize"]
    WORKER_TIME_TO_KILL = config_controller["workerTimeToKill"]
    MODEL_NUM_BLOCKS = config_controller["modelNumBlocks"]
    MODEL_NUM_CHANNELS = config_controller["modelNumChannels"]
    RESET_NETWORK = config_controller["resetNetwork"]
    LINEAR_WEIGHTING = config_controller["linearWeighting"]
    NUM_TRAIN_SAMPLES = config_controller["numTrainSamples"]
    NUM_SAVE_SAMPLES = config_controller["numSaveSamples"]
    BATCH_SIZE = config_controller["batchSize"]
    LR_INIT = config_controller["lrInit"]
    config_scheduler = config_controller["scheduler"]


RUN_NAME = f'{MODEL_NAME}_{MODEL_VARIANT}'

# rank = int(os.environ["SLURM_PROCID"])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
      f" {gpus_per_node} allocated GPUs per node.", flush=True)


logger = logging.getLogger()
logging.basicConfig(
    filename=f'logs/{RUN_NAME}_{os.environ["RANK"]}.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger.info(
    f'RUN_NAME = {RUN_NAME} RANK = {os.environ["RANK"]} alive at {time.time()}')


def setup_DDP():
    logger.info(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    logger.info(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    logger.info(f'WORLD_SIZE = {os.environ["WORLD_SIZE"]}')
    logger.info(f'RANK = {os.environ["RANK"]}')
    logger.info(f'LOCAL_RANK = {os.environ["LOCAL_RANK"]}')

    logger.info(f'cuda devices: {torch.cuda.device_count()}')

    # initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    assert int(os.environ["WORLD_SIZE"]) == WORLD_SIZE
    logger.info(
        f'Local Rank {os.environ["LOCAL_RANK"]} World Size {WORLD_SIZE}')


def cleanup():
    dist.destroy_process_group()


def json_to_scheduler(config_scheduler: dict, optimizer: optim.Optimizer
                      ) -> optim.lr_scheduler._LRScheduler:
    """
    Examples:
    {
        "name": "MultiStepLR",
        "kwargs": {
            "milestones": [30, 80],
            "gamma": 0.1
        }
    }
    {
        "name": "ReduceLROnPlateau",
        "kwargs": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 0.0001,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0,
            "eps": 1e-08,
            "verbose": True
        }
    }
    """
    which = {
        "MultiStepLR": optim.lr_scheduler.MultiStepLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    }
    return which[config_scheduler["name"]](optimizer,
                                           **config_scheduler["kwargs"])


epochify_time = 0
save_time = 0
trace_time = 0

model_kwargs = {
    "num_rows": NUM_ROWS,
    "num_cols": NUM_COLS,
    "action_size": ACTION_SIZE,
    "history_size": HISTORY_SIZE,
    "num_blocks": MODEL_NUM_BLOCKS,
    "num_channels": MODEL_NUM_CHANNELS
}

data_muncher_kwargs = {
    "num_worker_tasks": NUM_WORKER_TASKS,
    "num_groups": NUM_GROUPS,
    "run_name": BOOTSTRAP_NAME,
    "use_linear_wgt": LINEAR_WEIGHTING,
    "worker_ttk": WORKER_TIME_TO_KILL,
    "sync": SYNC,
    "num_train_samples": NUM_TRAIN_SAMPLES,
    "num_save_samples": NUM_SAVE_SAMPLES
}


def train_network(
        network: DDP, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
        iteration: int,
        state_tensor: torch.Tensor, distribution_tensor: torch.Tensor,
        outcome_tensor: torch.Tensor, timestamp_tensor: torch.Tensor):
    global epochify_time, save_time, trace_time

    num_samples = torch.tensor(state_tensor.shape[0], device=local_rank)
    logger.info(f"Rank {local_rank} has {num_samples.cpu().item()} samples.")

    # All-reduce the number of batches across all processes; only use the smallest number of batches
    torch.distributed.barrier()
    torch.distributed.all_reduce(num_samples, op=dist.ReduceOp.MIN)
    torch.distributed.barrier()

    num_samples = min(NUM_TRAIN_SAMPLES, num_samples.cpu().item())

    logger.info(
        f"Rank {local_rank} has {num_samples} samples after all-reduce.")

    state_tensor = state_tensor[-num_samples:]
    distribution_tensor = distribution_tensor[-num_samples:]
    outcome_tensor = outcome_tensor[-num_samples:]
    timestamp_tensor = timestamp_tensor[-num_samples:]

    dataset = TensorDataset(
        state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(10):
        epochify_time -= time.time()
        train_average_policy_loss, train_average_value_loss = epochify(iteration, 0, epoch,
                                                                       network, dataloader, optimizer, train=True)
        scheduler.step()
        dist.barrier()
        epochify_time += time.time()

        logger.info(
            f"{iteration}.{epoch} Rank {local_rank} " +
            f"Tr Pol: {train_average_policy_loss:.4f}, " +
            f"Tr Val: {train_average_value_loss:.4f}, "
        )

    trace_time -= time.time()
    if rank == 0 and iteration % 10 == 0:
        state_dict = network.module.state_dict()
        filepath = f"./data/models/{RUN_NAME}/" + \
            f"{RUN_NAME}_iteration_{iteration}.pt"
        trace_filepath = f"./data/models/{RUN_NAME}/" + \
            f"traced_{RUN_NAME}_iteration_{iteration}.pt"
        torch.save(state_dict, filepath)

        optimizer_filepath = f"./data/models/{RUN_NAME}/" + \
            f"{RUN_NAME}_optimizer_iteration_{iteration}.pt"
        torch.save(optimizer.state_dict(), optimizer_filepath)

        trace_model(filepath, torch.randn(1, 2 * HISTORY_SIZE + 1, NUM_ROWS, NUM_COLS),
                    trace_filepath, BasicGridNetwork, model_kwargs)
    dist.barrier()

    trace_time += time.time()

    logger.info(
        f"Epochify: {epochify_time:.2f}s, Save: {save_time:.2f}s, Trace: {trace_time:.2f}s")


def epochify(iteration: int, group: int, epoch: int,
             network: BasicGridNetwork, train_dataloader: DataLoader,
             optimizer: optim.Optimizer = None, train: bool = True, EPS: float = 1e-8) -> Tuple[float, float]:
    # logger.info(f"Rank {local_rank}:{world_size} entering epochify train={train}")
    if train:
        network.train()
    else:
        network.eval()

    total_policy_loss = 0.0  # These are turned into Tensors on epoch 0.
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

        total_policy_loss += policy_loss.detach()
        total_value_loss += value_loss.detach()
        num_batches += 1

    average_policy_loss = total_policy_loss / num_batches
    average_value_loss = total_value_loss / num_batches

    # Use distributed all reduce to average the losses across all processes.
    torch.distributed.barrier()
    torch.distributed.all_reduce(average_policy_loss,
                                 op=dist.ReduceOp.SUM)
    torch.distributed.all_reduce(average_value_loss,
                                 op=dist.ReduceOp.SUM)
    torch.distributed.barrier()

    average_value_loss /= world_size
    average_policy_loss /= world_size

    return average_policy_loss, average_value_loss


def main():
    setup_DDP()
    # Create the necessary directories
    os.makedirs(f"data/models/{RUN_NAME}", exist_ok=True)

    logger.info(f"Created necessary directories for {RUN_NAME}.")

    network = BasicGridNetwork(**model_kwargs).to(local_rank)

    network = DDP(network, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(network.parameters(), lr=LR_INIT)
    scheduler = json_to_scheduler(config_scheduler, optimizer)
    logger.info(
        f"Local rank {local_rank} network is on device {network.device}")

    muncher = data_muncher.DataMuncher(local_rank=local_rank, rank=rank, world_size=world_size,
                                       iter=0,
                                       **data_muncher_kwargs)

    for iteration in range(200):
        train_dataset = muncher.get()
        train_network(network, optimizer, scheduler,
                      iteration, **train_dataset)

    torch.distributed.barrier()
    logger.info(f"Rank {local_rank} finished training.")
    cleanup()


if __name__ == "__main__":
    main()
