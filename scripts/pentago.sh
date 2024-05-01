# See src/pentago_constants.py for setting parameters
LLsub ./pentago_master.sh [4,1,40] -g volta:2
LLsub ./pentago_worker.sh [8,48,1]