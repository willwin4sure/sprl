NUM_ITERS = 100
EXEC_PATH = "./cpp/build/Release/PTGSelfPlay.exe"

NUM_WORKER_TASKS = 384

# Parameters for each self-play iteration
NUM_GAMES_PER_WORKER = 2
UCT_ITERATIONS = 512
MAX_TRAVERSALS = 8
MAX_QUEUE_SIZE = 4

# Seed games at iteration 0
INIT_NUM_GAMES_PER_WORKER = 5
INIT_UCT_ITERATIONS = 32768
INIT_MAX_TRAVERSALS = 1
INIT_MAX_QUEUE_SIZE = 1

# Parameters for network and training
MODEL_NUM_BLOCKS = 2
MODEL_NUM_CHANNELS = 64
RESET_NETWORK = True
LINEAR_WEIGHTING = True
NUM_PAST_ITERS_TO_TRAIN = 100
MAX_GROUPS = 10
EPOCHS_PER_GROUP = 20
BATCH_SIZE = 1024
LR = 1e-3

RUN_NAME = f"iguana"
