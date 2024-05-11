#!/bin/sh

# See scripts/othello_controller.py for setting parameters
LLsub ./scripts/othello_controller.sh [1,1,40] -g volta:2
LLsub ./scripts/othello_worker.sh [8,48,1]