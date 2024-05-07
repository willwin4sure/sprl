/**
 * SelfPlay.cpp
 * 
 * Provides functionality to generate self-play data.
 */

#include "../games/GameState.hpp"
#include "../games/Game.hpp"

#include "../networks/Network.hpp"

#include "../random/Random.hpp"

#include "../tqdm/tqdm.hpp"

#include "../uct/UCTTree.hpp"

#include "../constants.hpp"

#include <iostream>
#include <string>

namespace SPRL {

/**
 * Generates self-play data using the given game and network, improved with UCT.
 * 
 * Returns a tuple of:
 * - A vector of states, where each state is a symmetrized version of the game state over time
 * - A vector of action distributions, where each distribution is a symmetrized version
 *   of the action distribution produced by UCT
 * - The outcome of the game (reward for the first player)
*/
template <int BOARD_SIZE, int ACTION_SIZE>
std::tuple<std::vector<GameState<BOARD_SIZE>>, std::vector<GameActionDist<ACTION_SIZE>>, float>
selfPlay(Game<BOARD_SIZE, ACTION_SIZE>* game,
         Network<BOARD_SIZE, ACTION_SIZE>* network,
         int numIters,
         int maxTraversals,
         int maxQueueSize,
         bool addNoise = true,
         bool symmetrizeNetwork = true,
         bool symmetrizeData = true) {

    using State = GameState<BOARD_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> states;
    std::vector<ActionDist> distributions;

    int numSymmetries = game->numSymmetries();
    std::vector<Symmetry> symmetries(numSymmetries);
    for (int i = 0; i < numSymmetries; ++i) {
        symmetries[i] = i;
    }

    State state = game->startState();
    UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state, addNoise, symmetrizeNetwork };

    int moveCount = 0;

    while (!state.isTerminal()) {
        if (symmetrizeData) {
            // Symmetrize the state and add to data
            std::vector<State> symmetrizedStates = game->symmetrizeState(state, symmetries);
            states.reserve(states.size() + symmetrizedStates.size());
            states.insert(states.end(), symmetrizedStates.begin(), symmetrizedStates.end());
        } else {
            states.push_back(state);
        }

        // ActionDist actionMask = game->actionMask(state);

        // std::cout << game->stateToString(state) << '\n';

        int iters = 0;
        while (iters < numIters) {
            auto [leaves, iter] = tree.searchAndGetLeaves(maxTraversals, maxQueueSize, network, EXPLORATION);

            if (leaves.size() > 0) {
                tree.evaluateAndBackpropLeaves(leaves, network);
            }

            iters += iter;
        }

        std::array<int32_t, ACTION_SIZE> visits = tree.getRoot()->getEdgeStatistics()->m_numberVisits;

        // Generate a PDF from these visit counts
        std::array<float, ACTION_SIZE> pdf{};

        {
            float totalVisits = 0.0f;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                totalVisits += visits[i];
            }

            float norm = 1.0f / totalVisits;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] = visits[i] * norm;
            }
        }

        // Raise it to 0.98f (temp ~ 1) if early game, else 10.0f (temp -> 0)
        if (moveCount < EARLY_GAME_CUTOFF) {
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] = std::pow(pdf[i], EARLY_GAME_EXP);
            }
        } else {
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] = std::pow(pdf[i], REST_GAME_EXP);
            }
        }

        // Renormalize
        {
            float total = 0.0f;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                total += pdf[i];
            }

            float norm = 1.0f / total;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] *= norm;
            }
        }

        // Generate a CDF from the PDF
        std::vector<float> cdf(ACTION_SIZE);

        {
            cdf[0] = pdf[0];
            for (int i = 1; i < ACTION_SIZE; ++i) {
                cdf[i] = cdf[i - 1] + pdf[i];
            }

            float norm = 1.0f / cdf[ACTION_SIZE - 1];
            for (int i = 0; i < ACTION_SIZE; ++i) {
                cdf[i] *= norm;
            }
        }

        if (symmetrizeData) {
            // Symmetrize the distributions and add to data
            std::vector<ActionDist> symmetrizedDists = game->symmetrizeActionDist(pdf, symmetries);
            distributions.reserve(distributions.size() + symmetrizedDists.size());
            distributions.insert(distributions.end(), symmetrizedDists.begin(), symmetrizedDists.end());
        } else {
            distributions.push_back(pdf);
        }

        // Sample from the CDF
        int action = GetRandom().SampleCDF(cdf);

        // Play the action by rerooting the tree and updating the state
        tree.rerootTree(action);
        state = game->nextState(state, action);

        ++moveCount;
    }

    float outcome = game->rewards(state).first;  // Reward for the first player

    return { states, distributions, outcome };
}

/**
 * Runs numGames many games and collects all their data. 
*/
template <int BOARD_SIZE, int ACTION_SIZE>
std::tuple<std::vector<GameState<BOARD_SIZE>>, std::vector<GameActionDist<ACTION_SIZE>>, std::vector<float>>
runIteration(Game<BOARD_SIZE, ACTION_SIZE>* game,
             Network<BOARD_SIZE, ACTION_SIZE>* network,
             int numGames, int numIters, int maxTraversals, int maxQueueSize) {

    using State = GameState<BOARD_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> allStates;
    std::vector<ActionDist> allDistributions;
    std::vector<float> allOutcomes;

    // auto pbar = tq::trange(numGames);
    // pbar.set_prefix("Generating self-play data: ");  // Do not change this prefix: Python side hooks into it
    for (int t = 0; t < numGames; ++t) {
        auto [states, distributions, outcome] = selfPlay(game, network, numIters, maxTraversals, maxQueueSize);

        allStates.reserve(allStates.size() + states.size());
        allStates.insert(allStates.end(), states.begin(), states.end());

        allDistributions.reserve(allDistributions.size() + distributions.size());
        allDistributions.insert(allDistributions.end(), distributions.begin(), distributions.end());

        for (int j = 0; j < states.size(); ++j) {
            if (states[j].getPlayer() == 0) {
                allOutcomes.push_back(outcome);
            } else {
                allOutcomes.push_back(-outcome);
            }
        }
        
        std::cout << t + 1 << " games played, " << allStates.size() << " states collected\n";
    }

    assert(allStates.size() == allDistributions.size());
    assert(allStates.size() == allOutcomes.size());

    return { allStates, allDistributions, allOutcomes };
}

} // namespace SPRL
