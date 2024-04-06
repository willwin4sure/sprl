#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl
echo "master thread"

python tests/robin.py -1 384
echo "done"