#!/bin/sh
cd ~/sprl
module load anaconda/2023a-pytorch

LLsub ./robin.sh [4,48,1]