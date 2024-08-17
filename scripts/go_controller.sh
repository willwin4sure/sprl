#!/bin/bash
#SBATCH --job-name=go_controller # This is just a name, doesn't matter.
#SBATCH --nodes=4 # This is supposed to be the total number of nodes.
#SBATCH --gres=gpu:volta:2 # Total number of gpus per node.
#SBATCH --ntasks-per-node=1 # I want one TASK per node; the task we're running *is* torchrun.
#SBATCH --cpus-per-task=20 # CPU Cores per task.
#SBATCH --distribution=cyclic:cyclic # This is the default, but it gets over-ridden by supercloud. So we have to set it here.
#SBATCH --output=/home/gridsan/rzhong/sprl/logs/%x-%j.out
#SBATCH --error=/home/gridsan/rzhong/sprl/logs/%x-%j.err


# zoom zoom 
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

echo "Running on host:: $RDZV_HOST"
module list

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO


module purge
module load anaconda/2023a-pytorch
module load nccl/2.18.1-cuda11.8 

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint "$RDZV_HOST:$RDZV_PORT" \
./scripts/go_controller.py