# Options

Here is a list of options which should be held in an Options struct.

1.  Worker Options:

    1. `NUM_GROUPS`, `NUM_WORKER_TASKS`: supercloud settings.
    2. `numIters`: Total number of iterations for the entire training process.
    3. `iterationOptions`: Options for the main training loop.
    4. `initIterationOptions`: Options for the initial iteration of the training loop.
    5. `model_name` The name of the model.
    6. `model_variant` The variant of the model, possibly the empty string.

2.  Iteration Options:

    1. `NUM_GAMES_PER_WORKER`: How many games to a worker should play out per iteration.
    2. `UCT_TRAVERSALS` How many traversals a worker should make at a root node when playing. This is called `numTraversals` internally.
    3. `MAX_BATCH_SIZE`, `MAX_QUEUE_SIZE`: Within a single `searchAndGetLeaves()` call, leaves are collected until there are `MAX_QUEUE_SIZE` leaves or `MAX_BATCH_SIZE` total traversals. Not all traversals return leaves.
    4. `EARLY_GAME_CUTOFF`, `EARLY_GAME_EXP`, `REST_GAME_EXP`, `FAST_PLAY_PROBABILITY`: Early game cutoff, early game exploration, rest game exploration, and fast play probability.
    5. `treeOptions`: Options for the tree.

3.  Tree Options:

    1. `addNoise`: If `addNoise` is true, `dirEps` of the priors on the root node will be Dirichlet noise with alpha parameter `dirAlpha`. (`dirEps` and `dirAlpha` are held in `NodeOptions`)
    2. `nodeOptions`: Options for the nodes.

4.  Node Options:

    1. `dirEps`, `dirAlpha`: If `addNoise` is true, `dirEps` of the priors on the root node will be Dirichlet noise with alpha parameter `dirAlpha`.
    2. `uWeight`: The weight of the U value in the UCT formula.
    3. `initQMethod`, `dropParent`: Child node expansion setting.
