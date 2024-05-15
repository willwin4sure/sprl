#!/bin/sh

module load anaconda/2023a-pytorch

# Initialize and Load Modules
cd ~/running_sims/sprl

echo "I am a worker process."
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

./cpp/build/RobinWorker $LLSUB_RANK $LLSUB_SIZE 31 \
    random 0 0 \
    ./data/models/gorilla_ablation_none/traced_gorilla_ablation_none_iteration_0.pt 0 0 \
    ./data/models/gorilla_ablation_none/traced_gorilla_ablation_none_iteration_5.pt 0 0 \
    ./data/models/gorilla_ablation_none/traced_gorilla_ablation_none_iteration_10.pt 0 0 \
    ./data/models/gorilla_ablation_none/traced_gorilla_ablation_none_iteration_15.pt 0 0 \
    ./data/models/gorilla_ablation_none/traced_gorilla_ablation_none_iteration_20.pt 0 0 \
    ./data/models/gorilla_ablation_only_linear/traced_gorilla_ablation_only_linear_iteration_0.pt 0 0 \
    ./data/models/gorilla_ablation_only_linear/traced_gorilla_ablation_only_linear_iteration_5.pt 0 0 \
    ./data/models/gorilla_ablation_only_linear/traced_gorilla_ablation_only_linear_iteration_10.pt 0 0 \
    ./data/models/gorilla_ablation_only_linear/traced_gorilla_ablation_only_linear_iteration_15.pt 0 0 \
    ./data/models/gorilla_ablation_only_linear/traced_gorilla_ablation_only_linear_iteration_20.pt 0 0 \
    ./data/models/gorilla_ablation_only_pq/traced_gorilla_ablation_only_pq_iteration_0.pt 0 1 \
    ./data/models/gorilla_ablation_only_pq/traced_gorilla_ablation_only_pq_iteration_5.pt 0 1 \
    ./data/models/gorilla_ablation_only_pq/traced_gorilla_ablation_only_pq_iteration_10.pt 0 1 \
    ./data/models/gorilla_ablation_only_pq/traced_gorilla_ablation_only_pq_iteration_15.pt 0 1 \
    ./data/models/gorilla_ablation_only_pq/traced_gorilla_ablation_only_pq_iteration_20.pt 0 1 \
    ./data/models/gorilla_ablation_only_reset/traced_gorilla_ablation_only_reset_iteration_0.pt 0 0 \
    ./data/models/gorilla_ablation_only_reset/traced_gorilla_ablation_only_reset_iteration_5.pt 0 0 \
    ./data/models/gorilla_ablation_only_reset/traced_gorilla_ablation_only_reset_iteration_10.pt 0 0 \
    ./data/models/gorilla_ablation_only_reset/traced_gorilla_ablation_only_reset_iteration_15.pt 0 0 \
    ./data/models/gorilla_ablation_only_reset/traced_gorilla_ablation_only_reset_iteration_20.pt 0 0 \
    ./data/models/gorilla_ablation_only_symm/traced_gorilla_ablation_only_symm_iteration_0.pt 1 0 \
    ./data/models/gorilla_ablation_only_symm/traced_gorilla_ablation_only_symm_iteration_5.pt 1 0 \
    ./data/models/gorilla_ablation_only_symm/traced_gorilla_ablation_only_symm_iteration_10.pt 1 0 \
    ./data/models/gorilla_ablation_only_symm/traced_gorilla_ablation_only_symm_iteration_15.pt 1 0 \
    ./data/models/gorilla_ablation_only_symm/traced_gorilla_ablation_only_symm_iteration_20.pt 1 0 \
    ./data/models/flamingo/traced_flamingo_iteration_0.pt 1 1 \
    ./data/models/flamingo/traced_flamingo_iteration_5.pt 1 1 \
    ./data/models/flamingo/traced_flamingo_iteration_10.pt 1 1 \
    ./data/models/flamingo/traced_flamingo_iteration_15.pt 1 1 \
    ./data/models/flamingo/traced_flamingo_iteration_20.pt 1 1 

echo "Done."
