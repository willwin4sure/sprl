#!/bin/sh

# Initialize and Load Modules
cd ~/sprl

echo "I am the controller process."

python ./scripts/go_controller.py

echo "Done."