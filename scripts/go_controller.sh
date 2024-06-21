#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl

echo "I am the controller process."

python ./scripts/go_controller.py

echo "Done."