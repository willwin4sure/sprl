#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/running_sims/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

./cpp/build/OTHWorker $LLSUB_RANK $LLSUB_SIZE

echo "Done."
