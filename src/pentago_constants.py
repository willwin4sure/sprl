# MAKE SURE THESE ARE IN SYNC WITH constants.hpp
# MAKE SURE THE RUN NAME IS IN SYNC WITH PTGWorker.cpp

NUM_ITERS = 100

NUM_GROUPS = 4
NUM_WORKER_TASKS = 384
WORKER_TIME_TO_KILL = 600

# Parameters for each self-play iteration
NUM_GAMES_PER_WORKER = 3
UCT_ITERATIONS = 16384
MAX_TRAVERSALS = 8
MAX_QUEUE_SIZE = 4

# Seed games at iteration 0
INIT_NUM_GAMES_PER_WORKER = 3
INIT_UCT_ITERATIONS = 524288
INIT_MAX_TRAVERSALS = 1
INIT_MAX_QUEUE_SIZE = 1

# Parameters for network and training
MODEL_NUM_BLOCKS = 3
MODEL_NUM_CHANNELS = 64
RESET_NETWORK = False
LINEAR_WEIGHTING = True
NUM_PAST_ITERS_TO_TRAIN = 10
MAX_GROUPS = 10
EPOCHS_PER_GROUP = 10
BATCH_SIZE = 1024
LR_INIT = 0.01
LR_DECAY_FACTOR = 0.1
LR_MILESTONE_ITERS = [10, 30, 60]

RUN_NAME = f"lion_prime"
