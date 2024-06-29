Here is a list of options which should be held in an Options struct.

1. Pretty information
    1. `model_name` The name of the model.
    2. `model_variant` The variant of the model, possibly the empty string.
2. High-level
    1. `NUM_GROUPS`, `NUM_WORKER_TASKS`: supercloud settings.
    2. `numIters`: Total number of iterations for the entire training process.
    3. `NUM_GAMES_PER_WORKER`: How many games to a worker should play out per iteration.
    4. `UCT_TRAVERSALS` How many traversals a worker should make at a root node when playing. This is called `numTraversals` internally.
    5. `INIT_NUM_GAMES_PER_WORKER`, `INIT_UCT_TRAVERSALS`: Initial variants
3. Inside UCT
    1. `MAX_BATCH_SIZE`, `MAX_QUEUE_SIZE`: Within a single `searchAndGetLeaves()` call, leaves are collected until there are `MAX_QUEUE_SIZE` leaves or `MAX_BATCH_SIZE` total traversals. Not all traversals return leaves.
    2. `INIT_MAX_BATCH_SIZE`, `INIT_MAX_QUEUE_SIZE`: Initial variants.
4. Inside UCTNode
    1. Dirichlet Noise
        1. `addNoise`, `dirEps`, `dirAlpha`: If `addNoise` is true, `dirEps` of the priors on the root node will be Dirichlet noise with alpha parameter `dirAlpha`.
    2. `initQMethod`, `dropParent`: Child node expansion setting.
    3. `symmetrizer`. This option will always be true, I see no reason to turn it off. Plus, it makes the file-saving and constexpr stuff easier.
