#!/bin/sh

# Initialize and Load Modules
cd ~/sprl

module load anaconda/2023a-pytorch

echo "I am the controller process."

python ./scripts/go_controller.py

echo "Done."