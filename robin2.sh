#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/running_sims/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE 7 \
    random 0 0 \
    ./data/models/manatee/traced_manatee_iteration_0.pt 1 1 \
    ./data/models/manatee/traced_manatee_iteration_3.pt 1 1 \
    ./data/models/manatee/traced_manatee_iteration_6.pt 1 1 \
    ./data/models/manatee/traced_manatee_iteration_9.pt 1 1 \
    ./data/models/manatee/traced_manatee_iteration_12.pt 1 1 \
    ./data/models/manatee/traced_manatee_iteration_15.pt 1 1 \

echo "Done."
