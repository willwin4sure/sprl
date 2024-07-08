"""
go_controller.py
"""

import json
import os
import sys
import tempfile
import time
from typing import List, Set, Tuple

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

startup_time -= time.time()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def show_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(
        f"Total (MiB): {t / (2 ** 20)}, Reserved: {r / (2 ** 20)}, Allocated: {a / (2 ** 20)}, Free: {f / (2 ** 20)}")


NUM_ROWS = 7
NUM_COLS = 7
ACTION_SIZE = 50
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
    # WORLD_SIZE = config_controller["worldSize"]
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


device = "cuda" if torch.cuda.is_available() else "cpu"


def setup(device: int, world_size: int):
    # This needs to be changed if we are using multiple machines.
    os.environ['MASTER_ADDR'] = 'localhost'
    # This can be any number.
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", device=device, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def collate_data(iteration: int, live_workers: Set[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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


worker_seen = [0 for i in range(NUM_WORKER_TASKS)]


def scoop_data(iteration: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    For each worker, scoop up all data that has not already been scooped up.
    """

    global worker_seen

    # in the case where iteration is 0, only, wait until every single worker has completed at least one game.
    if iteration == 0:
        print("Waiting for all workers to have data...")
        while True:
            all_workers_have_data = True
            for task_id in range(NUM_WORKER_TASKS):
                group = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)

                thread_save_path = f"data/games/{RUN_NAME}/{group}/{task_id}"

                if not (os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_states.npy") and
                        os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_distributions.npy") and
                        os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_outcomes.npy")):
                    all_workers_have_data = False
                    break

            if all_workers_have_data:
                break

            time.sleep(10)

    new_states = []
    new_distributions = []
    new_outcomes = []
    new_timestamps = []

    start_time = None

    for task_id in range(NUM_WORKER_TASKS):
        group = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)

        thread_save_path = f"data/games/{RUN_NAME}/{group}/{task_id}"

        while True:
            if (os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_states.npy") and
                os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_distributions.npy") and
                    os.path.exists(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_outcomes.npy")):

                states = torch.Tensor(
                    np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_states.npy"))
                distributions = torch.Tensor(np.load(
                    f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_distributions.npy"))
                outcomes = torch.Tensor(
                    np.load(f"{thread_save_path}/{RUN_NAME}_iteration_{worker_seen[task_id]}_outcomes.npy"))
                timestamps = torch.Tensor(
                    [iteration + 1 if LINEAR_WEIGHTING else 1 for _ in range(states.shape[0])])

                assert states.shape[0] == distributions.shape[0] == outcomes.shape[0] == timestamps.shape[0]

                new_states.append(states)
                new_distributions.append(distributions)
                new_outcomes.append(outcomes)
                new_timestamps.append(timestamps)

                worker_seen[task_id] += 1
            else:
                break

    print(
        f"Total samples for iteration {iteration}: {sum([s.shape[0] for s in new_states])}")

    print(f"Total scooped samples from each worker:")
    print(" ".join(str(i) for i in worker_seen))

    return new_states, new_distributions, new_outcomes, new_timestamps


epochify_time = 0
save_time = 0
trace_time = 0


def train_network(network: BasicGridNetwork, learning_rate: float, iteration: int,
                  state_tensor: torch.Tensor, distribution_tensor: torch.Tensor,
                  outcome_tensor: torch.Tensor, timestamp_tensor: torch.Tensor):
    global epochify_time, save_time, trace_time
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
                epochify_time -= time.time()
                train_average_policy_loss, train_average_value_loss = epochify(
                    network, train_dataloader, optimizer, train=True)

                val_average_policy_loss, val_average_value_loss = epochify(
                    network, val_dataloader, train=False)
                epochify_time += time.time()
                save_time -= time.time()

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


def epochify(network: BasicGridNetwork, train_dataloader: DataLoader,
             optimizer: optim.Optimizer = None, train: bool = True, EPS: float = 1e-8) -> Tuple[float, float]:
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


collation_time = 0
mem_time = 0
cat_time = 0
train_time = 0


def main():
    global collation_time, mem_time, cat_time, train_time
    print(f"I have access to {device}.")

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

    all_state_tensors: List[torch.Tensor] = []
    all_distribution_tensors: List[torch.Tensor] = []
    all_outcome_tensors: List[torch.Tensor] = []
    all_timestamp_tensors: List[torch.Tensor] = []

    network = BasicGridNetwork(
        NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)
    learning_rate = LR_INIT

    live_workers = set(range(NUM_WORKER_TASKS))

    print(f"Startup time = {time.time() - startup_time:.2f}s")

    for iteration in range(NUM_ITERS):
        collation_time -= time.time()
        print(f"Starting iteration {iteration}...")

        if iteration in LR_MILESTONE_ITERS:
            print(f"Decaying learning rate to {learning_rate}.")
            learning_rate *= LR_DECAY_FACTOR

        if RESET_NETWORK:
            print(f"Resetting network.")
            network = BasicGridNetwork(
                NUM_ROWS, NUM_COLS, ACTION_SIZE, HISTORY_SIZE, MODEL_NUM_BLOCKS, MODEL_NUM_CHANNELS)

        if SYNC:
            new_states, new_distributions, new_outcomes, new_timestamps = collate_data(
                iteration, live_workers)

            # Do not push anybody to GPU yet, before we clear out the old data from vram.
            new_state_tensor = torch.cat(new_states, dim=0)
            new_distribution_tensor = torch.cat(
                new_distributions, dim=0)
            new_outcome_tensor = torch.cat(
                new_outcomes, dim=0).unsqueeze(1)
            new_timestamp_tensor = torch.cat(
                new_timestamps, dim=0).unsqueeze(1)

            assert new_state_tensor.shape[0] == new_distribution_tensor.shape[0] \
                == new_outcome_tensor.shape[0] == new_timestamp_tensor.shape[0]

            # When sync, all games in a single iteration are a single tensor.
            # NUM_PAST_ITERS_TO_TRAIN is the number of iterations to keep.
            all_state_tensors.append(new_state_tensor)
            all_distribution_tensors.append(new_distribution_tensor)
            all_outcome_tensors.append(new_outcome_tensor)
            all_timestamp_tensors.append(new_timestamp_tensor)
        else:
            new_states, new_distributions, new_outcomes, new_timestamps = scoop_data(
                iteration)

            # When async, each game is a separate tensor.
            # NUM_PAST_ITERS_TO_TRAIN is the number of games to keep.
            all_state_tensors.extend(new_states)
            all_distribution_tensors.extend(new_distributions)
            all_outcome_tensors.extend(new_outcomes)
            all_timestamp_tensors.extend(new_timestamps)
        collation_time += time.time()
        mem_time -= time.time()
        assert (len(all_state_tensors) == len(all_distribution_tensors)
                == len(all_outcome_tensors) == len(all_timestamp_tensors))

        while len(all_state_tensors) > NUM_PAST_ITERS_TO_TRAIN:
            del all_state_tensors[0]
            del all_distribution_tensors[0]
            del all_outcome_tensors[0]
            del all_timestamp_tensors[0]
        #     s = all_state_tensors.pop(0).to('cpu')
        #     d = all_distribution_tensors.pop(0).to('cpu')
        #     o = all_outcome_tensors.pop(0).to('cpu')
        #     t = all_timestamp_tensors.pop(0).to('cpu')
        #     s.detach()
        #     s.grad = None
        #     d.detach()
        #     d.grad = None
        #     o.detach()
        #     o.grad = None
        #     t.detach()
        #     t.grad = None
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        mem_time += time.time()
        cat_time -= time.time()

        # Push everything to gpu.
        # all_state_tensors = [s.to(device) for s in all_state_tensors]
        # all_distribution_tensors = [d.to(device)
        #                             for d in all_distribution_tensors]
        # all_outcome_tensors = [o.to(device) for o in all_outcome_tensors]
        # all_timestamp_tensors = [t.to(device)
        #                          for t in all_timestamp_tensors]

        train_state_tensor = torch.cat(all_state_tensors, dim=0).to(device)
        train_distribution_tensor = torch.cat(
            all_distribution_tensors, dim=0).to(device)
        train_outcome_tensor = torch.cat(all_outcome_tensors, dim=0).to(device)
        train_timestamp_tensor = torch.cat(
            # - max(0, iteration + 1 - NUM_PAST_ITERS_TO_TRAIN)
            all_timestamp_tensors, dim=0).to(device)

        # assert that all tensors are on cpu.

        assert all([s.device == torch.device(type='cpu')
                   for s in all_state_tensors])
        assert all([d.device == torch.device(type='cpu')
                   for d in all_distribution_tensors])
        assert all([o.device == torch.device(type='cpu')
                   for o in all_outcome_tensors])
        assert all([t.device == torch.device(type='cpu')
                   for t in all_timestamp_tensors])

        assert train_state_tensor.shape[0] == train_distribution_tensor.shape[0]\
            == train_outcome_tensor.shape[0] == train_timestamp_tensor.shape[0]
        assert torch.min(train_timestamp_tensor) > 0

        cat_time += time.time()
        train_time -= time.time()
        train_network(network, learning_rate, iteration,
                      train_state_tensor,
                      train_distribution_tensor, train_outcome_tensor, train_timestamp_tensor)
        train_time += time.time()

        cat_time -= time.time()

        # Take everything off gpu, manually.
        train_state_tensor = train_state_tensor.to('cpu')
        train_distribution_tensor = train_distribution_tensor.to('cpu')
        train_outcome_tensor = train_outcome_tensor.to('cpu')
        train_timestamp_tensor = train_timestamp_tensor.to('cpu')
        # all_state_tensors = [s.to('cpu') for s in all_state_tensors]
        # all_distribution_tensors = [d.to('cpu')
        #                             for d in all_distribution_tensors]
        # all_outcome_tensors = [o.to('cpu') for o in all_outcome_tensors]
        # all_timestamp_tensors = [t.to('cpu') for t in all_timestamp_tensors]

        # wipe the gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        # Show wiped memory
        print("Wiped memory.")
        show_memory()
        cat_time += time.time()

        print(
            f"Col: {collation_time:.2f}s, Mem: {mem_time:.2f}s, Cat: {cat_time:.2f}s, Tr: {train_time:.2f}s")


if __name__ == "__main__":
    # mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    main()
