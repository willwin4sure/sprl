/**
 * selfPlay.cpp
 *
 * This file compiles to an executable that can be used to generated
 * self-play data, where an input argument is the file path of the
 * traced PyTorch model, and other inputs include the number of
 * search iterations and the batch size for the search.
 *
 * It is written into a `.npy` format that is readable by the Python
 * training loop.
 */

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/ConnectFourNetwork.hpp"

#include "random/Random.hpp"

#include "tqdm/tqdm.hpp"

#include "uct/UCTTree.hpp"

#include "constants.hpp"

#include <iostream>
#include <string>

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
std::tuple<std::vector<SPRL::GameState<BOARD_SIZE>>, std::vector<SPRL::GameActionDist<ACTION_SIZE>>, float>
selfPlay(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game,
         SPRL::Network<BOARD_SIZE, ACTION_SIZE>* network,
         int numIters,
         int maxTraversals,
         int maxQueueSize,
         bool addNoise = true,
         bool symmetrizeNetwork = true,
         bool symmetrizeData = true) {

    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    std::vector<State> states;
    std::vector<ActionDist> distributions;

    int numSymmetries = game->numSymmetries();
    std::vector<SPRL::Symmetry> symmetries(numSymmetries);
    for (int i = 0; i < numSymmetries; ++i) {
        symmetries[i] = i;
    }

    State state = game->startState();
    SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state, addNoise, symmetrizeNetwork };

    int moveCount = 0;

    while (!game->isTerminal(state)) {
        if (symmetrizeData) {
            // Symmetrize the state and add to data
            std::vector<State> symmetrizedStates = game->symmetrizeState(state, symmetries);
            states.reserve(states.size() + symmetrizedStates.size());
            states.insert(states.end(), symmetrizedStates.begin(), symmetrizedStates.end());
        } else {
            states.push_back(state);
        }

        ActionDist actionMask = game->actionMask(state);

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
        if (moveCount < 10) {
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] = std::pow(pdf[i], 0.98f);
            }
        } else {
            for (int i = 0; i < ACTION_SIZE; ++i) {
                pdf[i] = std::pow(pdf[i], 10.0f);
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
        int action = SPRL::GetRandom().SampleCDF(cdf);

        // Play the action by rerooting the tree and updating the state
        tree.rerootTree(action);
        state = game->nextState(state, action);

        ++moveCount;
    }

    float outcome = game->rewards(state).first;  // reward for the first player

    return { states, distributions, outcome };
}

/**
 * Runs numGames many games and collects all their data. 
*/
template <int BOARD_SIZE, int ACTION_SIZE>
std::tuple<std::vector<SPRL::GameState<BOARD_SIZE>>, std::vector<SPRL::GameActionDist<ACTION_SIZE>>, std::vector<float>>
runIteration(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game,
             SPRL::Network<BOARD_SIZE, ACTION_SIZE>* network,
             int numGames, int numIters, int maxTraversals, int maxQueueSize) {

    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    std::vector<State> allStates;
    std::vector<ActionDist> allDistributions;
    std::vector<float> allOutcomes;

    auto pbar = tq::trange(numGames);
    pbar.set_prefix("Generating self-play data: ");
    for (int t : pbar) {
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
        
        pbar << " " << t << " " << allStates.size();
    }

    assert(allStates.size() == allDistributions.size());
    assert(allStates.size() == allOutcomes.size());

    return { allStates, allDistributions, allOutcomes };
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./selfPlay.exe <model_path> <save_path> <numGames> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath { argv[1] };
    std::string savePath { argv[2] };
    int numGames = std::stoi(argv[3]);
    int numIters = std::stoi(argv[4]);
    int maxTraversals = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);

    SPRL::ConnectFour game;

    SPRL::Network<42, 7>* network;

    SPRL::RandomNetwork<42, 7> randomNetwork {};
    SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    if (modelPath == "random") {
        std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    auto [states, distributions, outcomes] = runIteration(&game, network, numGames, numIters, maxTraversals, maxQueueSize);

    std::vector<float> embeddedStates;
    for (const SPRL::GameState<42>& state : states) {
        for (int player = 0; player < 2; ++player) {
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 7; ++col) {
                    if (state.getBoard()[row * 7 + col] == player) {
                        embeddedStates.push_back(1.0f);
                    } else {
                        embeddedStates.push_back(0.0f);
                    }
                }
            }
        }

        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 7; ++col) {
                embeddedStates.push_back((state.getPlayer() == 0) ? 1.0f : 0.0f);
            }
        }
    }

    npy::npy_data_ptr<float> stateData {};
    stateData.data_ptr = embeddedStates.data();
    stateData.shape = { static_cast<unsigned long>(states.size()), 3, 6, 7 };

    npy::write_npy(savePath + "_states.npy", stateData);

    std::vector<float> embeddedDistributions;
    for (const SPRL::GameActionDist<7>& dist : distributions) {
        for (int i = 0; i < 7; ++i) {
            embeddedDistributions.push_back(dist[i]);
        }
    }

    npy::npy_data_ptr<float> distData {};
    distData.data_ptr = embeddedDistributions.data();
    distData.shape = { static_cast<unsigned long>(distributions.size()), 7 };

    npy::write_npy(savePath + "_distributions.npy", distData);

    npy::npy_data_ptr<float> outcomeData {};
    outcomeData.data_ptr = outcomes.data();
    outcomeData.shape = { static_cast<unsigned long>(outcomes.size()) };

    npy::write_npy(savePath + "_outcomes.npy", outcomeData);

    return 0;
}   
