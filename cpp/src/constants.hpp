#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

constexpr int SEED = 0;

constexpr float EXPLORATION = 1.0f;

constexpr float DIRICHLET_EPSILON = 0.25f;
constexpr float DIRICHLET_ALPHA = 0.1f;

constexpr int EARLY_GAME_CUTOFF = 20;
constexpr float EARLY_GAME_EXP = 0.98f;
constexpr float REST_GAME_EXP = 10.0f;

constexpr int NUM_ITERS = 100;

constexpr int NUM_GROUPS = 4;
constexpr int NUM_WORKER_TASKS = 384;

constexpr int NUM_GAMES_PER_WORKER = 3;
constexpr int UCT_ITERATIONS = 8192;
constexpr int MAX_TRAVERSALS = 8;
constexpr int MAX_QUEUE_SIZE = 4;

constexpr int INIT_NUM_GAMES_PER_WORKER = 3;
constexpr int INIT_UCT_ITERATIONS = 131072;
constexpr int INIT_MAX_TRAVERSALS = 1;
constexpr int INIT_MAX_QUEUE_SIZE = 1;

#endif
