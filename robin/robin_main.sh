#!/bin/bash
cd ~/sprl
module load anaconda/2023a-pytorch

LLsub ./robin/robin.sh [3,48,1]