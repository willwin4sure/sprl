#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python tests/robin.py $LLSUB_RANK $LLSUB_SIZE

echo "done"