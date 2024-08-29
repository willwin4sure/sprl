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


./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE \
    quail 3 \
    random 1 0 \
    quail_gamma 46 \
    0 1 2 3 4 5 \
    10 20 30 40 50 60 70 80 90 100 \
    110 120 130 140 150 160 170 180 190 200 \
    210 220 230 240 250 260 270 280 290 300 \
    310 320 330 340 350 360 366 367 368 369 \
    quail_delta 15 \
    0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 \
    # panda_gamma_slower 28 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 100 \
    # 120 140 160 180 189 \
    # quail_ddp 19 \
    # 0 1 2 3 4 5 \
    # 10 20 30 40 50 60 70 80 90 100 \
    # 105 110 114 \
    # panda_gamma_fast 28 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 100 \
    # 120 140 160 180 199 \
    # panda_delta_replicate_fast_new_prime 23 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 99 \
    # panda_delta_replicate_slow_new_prime 23 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 99 \
    # panda_delta_async_slow_2400 23 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 97 \
    # panda_epsilon_async_fast 36 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 100 \
    # 120 140 160 180 200 \
    # 225 250 275 300 325 \
    # 350 375 399 \
    # panda_epsilon_async_slow_mm 36 \
    # 0 1 2 3 4 5 6 7 8 9 10 \
    # 20 25 30 35 40 45 50 \
    # 60 70 80 90 100 \
    # 120 140 160 180 200 \
    # 225 250 275 300 325 \
    # 350 375 389 \

    
echo "Done."