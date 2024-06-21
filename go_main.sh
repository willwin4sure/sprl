#!/bin/sh

# See scripts/go_controller.py for setting parameters
LLsub ./scripts/go_controller.sh [1,1,40] -g volta:2
LLsub ./scripts/go_worker.sh [8,48,1]