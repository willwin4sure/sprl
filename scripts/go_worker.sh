#!/bin/sh

# Initialize and Load Modules
cd ~/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

./cpp/build/GoWorker $LLSUB_RANK $LLSUB_SIZE

echo "Done."