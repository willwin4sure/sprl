#!/bin/bash

module load anaconda/2023a-pytorch

sbatch ./scripts/go_controller.sh
LLsub ./scripts/go_worker.sh [8,48,1] --name=GoWorker --time=4-4:00:00

watch -n 0.1 LLstat