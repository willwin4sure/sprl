#!/bin/bash

module load anaconda/2023a-pytorch

sbatch ./scripts/go_controller.sh
LLsub ./scripts/go_worker.sh [2,48,1]