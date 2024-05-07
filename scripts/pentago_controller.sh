#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/running_sims/sprl

echo "I am the controller process."

python ./scripts/pentago_controller.py

echo "Done."