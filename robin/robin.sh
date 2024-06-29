#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE 40 \
    random \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_0.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_10.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_20.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_30.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_40.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_50.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_60.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_70.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_80.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_90.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_100.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_110.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_120.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_130.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_140.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_150.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_160.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_170.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_180.pt \
    ./data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_190.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_0.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_10.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_20.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_30.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_40.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_50.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_60.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_70.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_80.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_90.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_100.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_110.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_120.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_130.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_140.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_150.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_160.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_170.pt \
    ./data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_180.pt \

echo "Done."

echo "Starting robin.py"
python robin.py