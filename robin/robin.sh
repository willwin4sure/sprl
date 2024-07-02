#!/bin/bash

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

# Format:
# ./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE (TOURNAMENT NAME) (NUM_TEAMS) \
#    (TEAM_NAME_1) (NUM_PLAYERS_1) (ITERATION_1) (ITERATION_2) ... (ITERATION_N) \
#    (TEAM_NAME_2) (NUM_PLAYERS_2) (ITERATION_1) (ITERATION_2) ... (ITERATION_N) \
#    ...


./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE yottapanda 5 \
    random 1 0\
    panda_gamma_fast 8 0 20 40 60 80 100 150 199 \
    panda_gamma_slower 8 0 20 40 60 80 100 150 189 \
    panda_delta_replicate_fast_new_prime 6 0 20 40 60 80 99 \
    panda_delta_replicate_slow_new_prime 6 0 20 40 60 80 99 \

echo "Done."

echo "Starting robin.py"
python ./robin/robin.py