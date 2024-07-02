#!/bin/bash
cd ~/sprl
module load anaconda/2023a-pytorch

LLsub ./robin/robin.sh [4,48,1]