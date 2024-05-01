#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl
echo "controller thread"

python scripts/pentago_controller.py

echo "done"
