#!/bin/sh

# See src/pentago_constants.py for setting parameters
LLsub ./scripts/pentago_controller.sh [1,1,40] -g volta:2
LLsub ./scripts/pentago_worker.sh [8,48,1]