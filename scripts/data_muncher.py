"""
go_controller.py
"""

import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, List, Set, Tuple

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

logger = logging.getLogger()


def show_memory(local_rank):
    t = torch.cuda.get_device_properties(local_rank).total_memory
    r = torch.cuda.memory_reserved(local_rank)
    a = torch.cuda.memory_allocated(local_rank)
    f = r-a  # free inside reserved
    logger.info(
        f"local_rank {local_rank} Total (MiB): {t / (2 ** 20)}, Reserved: {r / (2 ** 20)}, Allocated: {a / (2 ** 20)}, Free: {f / (2 ** 20)}")


class DataMuncher():
    def __init__(self,
                 local_rank, rank,
                 world_size, iteration,
                 num_worker_tasks, num_groups, run_name,
                 linear_weighting, worker_time_to_kill, sync,
                 num_train_samples,
                 num_save_samples,
                 ):

        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.num_worker_tasks = num_worker_tasks
        self.live_workers = set(
            list(range(rank, num_worker_tasks, world_size))
        )
        self.worker_seen = [0 for i in range(num_worker_tasks)]

        self.my_workers = list(range(rank, num_worker_tasks, world_size))

        self.num_groups = num_groups
        self.run_name = run_name

        self.linear_weighting = linear_weighting
        self.worker_time_to_kill = worker_time_to_kill

        self.all_state_tensors = []
        self.all_distribution_tensors = []
        self.all_outcome_tensors = []
        self.all_timestamp_tensors = []

        self.sync = sync
        self.iteration = iteration
        self.num_train_samples = num_train_samples
        self.num_save_samples = num_save_samples

    def sync_collate(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        new_states = []
        new_distributions = []
        new_outcomes = []
        new_timestamps = []

        start_time = None

        finished_workers = set()

        while True:
            for task_id in self.live_workers:
                group = task_id // (self.num_worker_tasks // self.num_groups)

                thread_save_path = f"data/games/" + \
                    f"{self.run_name}/{group}/{task_id}"

                # Check if the worker is unfinished and has saved its data.
                if (not task_id in finished_workers and
                    os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_states.npy") and
                    os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_distributions.npy") and
                        os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_outcomes.npy")):

                    # Worker has finished collecting its data.
                    finished_workers.add(task_id)

                    states = torch.Tensor(
                        np.load(f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_states.npy"))
                    distributions = torch.Tensor(np.load(
                        f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_distributions.npy"))
                    outcomes = torch.Tensor(
                        np.load(f"{thread_save_path}/{self.run_name}_iteration_{self.iteration}_outcomes.npy"))
                    timestamps = torch.Tensor(
                        [self.iteration + 1 if self.linear_weighting else 1 for _ in range(states.shape[0])])

                    assert states.shape[0] == distributions.shape[0] == outcomes.shape[0] == timestamps.shape[0]

                    new_states.append(states)
                    new_distributions.append(distributions)
                    new_outcomes.append(outcomes)
                    new_timestamps.append(timestamps)

            if start_time is None and len(self.live_workers) > len(finished_workers) > len(self.live_workers) // 2:
                # Over half of the workers have finished, start a timer after which we will kill the rest
                logger.info(
                    f"Over half of the workers have finished. Starting timer to kill the rest.")
                start_time = time.time()

            if start_time is not None and time.time() - start_time > self.worker_time_to_kill:
                # Kill the remaining workers after time has expired
                for task_id in self.live_workers - finished_workers:
                    self.live_workers.remove(task_id)

                logger.info(
                    f"Killing unfinished workers: {len(self.live_workers)} left.")
                break

            if len(finished_workers) == len(self.live_workers):
                # All workers have finished
                break

            logger.info(
                f"Spinning on workers to finish... {len(finished_workers)} / {len(self.live_workers)} are complete.")
            time.sleep(30)

        logger.info(
            f"Total samples for iteration {self.iteration}: {sum([s.shape[0] for s in new_states])}")

        return new_states, new_distributions, new_outcomes, new_timestamps

    def async_collate(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        For each worker, scoop up all data that has not already been scooped up.
        """
        # in the case where iteration is 0, only, wait until every single worker has completed at least one game.
        if self.iteration == 0:
            while True:
                all_workers_have_data = True
                # Not just my_workers; I want all workers under all masters to have finished at least one game.
                for task_id in range(self.num_worker_tasks):
                    group = task_id // (self.num_worker_tasks //
                                        self.num_groups)

                    thread_save_path = f"data/games/" + \
                        f"{self.run_name}/{group}/{task_id}"

                    if not (os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_states.npy") and
                            os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_distributions.npy") and
                            os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_outcomes.npy")):
                        all_workers_have_data = False
                        logger.info(
                            f"Waiting for all workers to have data (missing {task_id})")

                        break

                if all_workers_have_data:
                    break

                time.sleep(10)

        new_states = []
        new_distributions = []
        new_outcomes = []
        new_timestamps = []

        start_time = None

        for task_id in self.my_workers:
            group = task_id // (self.num_worker_tasks // self.num_groups)

            thread_save_path = f"data/games/{self.run_name}/{group}/{task_id}"

            while True:
                if (os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_states.npy") and
                    os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_distributions.npy") and
                        os.path.exists(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_outcomes.npy")):

                    states = torch.Tensor(
                        np.load(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_states.npy"))
                    distributions = torch.Tensor(np.load(
                        f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_distributions.npy"))
                    outcomes = torch.Tensor(
                        np.load(f"{thread_save_path}/{self.run_name}_iteration_{self.worker_seen[task_id]}_outcomes.npy"))
                    timestamps = torch.Tensor(
                        [self.iteration + 1 if self.linear_weighting else 1 for _ in range(states.shape[0])])

                    assert states.shape[0] == distributions.shape[0] == outcomes.shape[0] == timestamps.shape[0]

                    new_states.append(states)
                    new_distributions.append(distributions)
                    new_outcomes.append(outcomes)
                    new_timestamps.append(timestamps)

                    self.worker_seen[task_id] += 1
                else:
                    break

        logger.info(
            f"Total new samples for iteration {self.iteration}: {sum([s.shape[0] for s in new_states])}")

        logger.info(f"Total scooped samples from each worker:")
        logger.info(" ".join(str(i) for i in self.worker_seen))

        return new_states, new_distributions, new_outcomes, new_timestamps

    def get(self):
        if self.sync:
            new_states, new_distributions, new_outcomes, new_timestamps = self.sync_collate()
        else:
            new_states, new_distributions, new_outcomes, new_timestamps = self.async_collate()

        # Do not push anybody to GPU yet, before we clear out the old data from vram.
        new_outcomes = [o.unsqueeze(1) for o in new_outcomes]
        new_timestamps = [t.unsqueeze(1) for t in new_timestamps]

        self.all_state_tensors.extend(new_states)
        self.all_distribution_tensors.extend(new_distributions)
        self.all_outcome_tensors.extend(new_outcomes)
        self.all_timestamp_tensors.extend(new_timestamps)

        logger.info(f"Before truncating, the total number of states is:" +
                    f"{len(self.all_state_tensors)}")
        logger.info(f"Before truncating, the total number of samples is:" +
                    f"{sum([s.shape[0] for s in self.all_state_tensors])}")

        assert (len(self.all_state_tensors) == len(self.all_distribution_tensors)
                == len(self.all_outcome_tensors) == len(self.all_timestamp_tensors))

        num_samples = sum([s.shape[0] for s in self.all_state_tensors])

        while num_samples > self.num_save_samples:
            num_samples -= self.all_state_tensors[0].shape[0]
            del self.all_state_tensors[0]
            del self.all_distribution_tensors[0]
            del self.all_outcome_tensors[0]
            del self.all_timestamp_tensors[0]

        logger.info(f"After truncating, the total number of states is:" +
                    f"{len(self.all_state_tensors)}")
        logger.info(f"After truncating, the total number of samples is:" +
                    f"{sum([s.shape[0] for s in self.all_state_tensors])}")

        # Our objective here is to get a subset of the data that contains exactly num_train_samples samples.
        # We first over-shoot, then we truncate.
        index_permutation = np.random.permutation(len(self.all_state_tensors))
        train_cutoff = 0
        train_samples = 0
        while train_samples < self.num_train_samples and train_cutoff < len(index_permutation):
            train_samples += self.all_state_tensors[index_permutation[train_cutoff]].shape[0]
            train_cutoff += 1

        logger.info(f"Total number of games used for training: {train_cutoff}")
        logger.info(f"Total number of samples used for training: " +
                    f"{train_samples}")

        index_subset = index_permutation[:train_cutoff]
        train_state_tensor = torch.cat(
            [self.all_state_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_distribution_tensor = torch.cat(
            [self.all_distribution_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_outcome_tensor = torch.cat(
            [self.all_outcome_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_timestamp_tensor = torch.cat(
            [self.all_timestamp_tensors[i] for i in index_subset], dim=0).to(self.local_rank)

        train_timestamp_tensor = train_timestamp_tensor - \
            torch.min(train_timestamp_tensor) + 1

        show_memory(self.local_rank)
        self.iteration += 1
        logger.info("Pushed data to GPU.")
        return {
            "state_tensor": train_state_tensor,
            "distribution_tensor": train_distribution_tensor,
            "outcome_tensor": train_outcome_tensor,
            "timestamp_tensor": train_timestamp_tensor
        }
