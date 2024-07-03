#!/bin/bash
cd ~/sprl
module load anaconda/2023a-pytorch
for i in {0..47}
do
    # This will run 48 workers in parallel
    # Both the output and error streams are redirected to a file
    ./cpp/build/RobinWorker $i 48 yottapanda 5 \
        random 1 0\
        panda_gamma_fast 8 0 20 40 60 80 100 150 199 \
        panda_gamma_slower 8 0 20 40 60 80 100 150 189 \
        panda_delta_replicate_fast_new_prime 6 0 20 40 60 80 99 \
        panda_delta_replicate_slow_new_prime 6 0 20 40 60 80 99 \
        > ./ROBIN_OUTPUT/robin_worker_$i.txt 2>&1 &
done
