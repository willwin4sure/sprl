#ifndef SPRL_SELF_PLAY_HPP
#define SPRL_SELF_PLAY_HPP

/**
 * @file SelfPlay.cpp
 * 
 * Provides functionality to generate self-play data.
 */

#include "../games/GameNode.hpp"
#include "../networks/INetwork.hpp"
#include "../symmetry/ISymmetrizer.hpp"
#include "../uct/UCTTree.hpp"

#include "../utils/random.hpp"
#include "../utils/tqdm.hpp"

#include "../constants.hpp"

#include <iostream>
#include <string>

namespace SPRL {

/**
 * Generates a single game of self-play data using the given game and network, improved with UCT.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param rootNode The root node of the game tree.
 * @param network The neural network to use for evaluation.
 * @param numTraversals The number of UCT traversals to run per move.
 * @param maxBatchSize The maximum number of traversals per batch of search.
 * @param maxQueueSize The maximum number of states to evaluate per batch of search.
 * @param dirEps The epsilon value for the Dirichlet noise.
 * @param dirAlpha The alpha value for the Dirichlet noise.
 * @param initQMethod The method to initialize the Q values.
 * @param dropParent Whether to drop the parent network evaluation from the Q values.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data (or nullptr).
 * @param addNoise Whether to add Dirichlet noise to the root node.
 * 
 * @returns A tuple of:
 *     1. A vector of states, where each state is a symmetrized version of the game state over time.
 *     2. A vector of action distributions, where each distribution is a symmetrized version
 *        of the action distribution produced by UCT.
 *     3. A vector of outcomes, where each outcome is the reward for the corresponding player
 *        that took an action at any given state.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
std::tuple<std::vector<State>, std::vector<GameActionDist<ACTION_SIZE>>, std::vector<Value>>
selfPlay(std::unique_ptr<GameNode<ImplNode, State, ACTION_SIZE>> rootNode,
         INetwork<State, ACTION_SIZE>* network,
         int numTraversals, int maxBatchSize, int maxQueueSize,
         float dirEps, float dirAlpha, InitQ initQMethod, bool dropParent = true,
         ISymmetrizer<State, ACTION_SIZE>* symmetrizer = nullptr, bool addNoise = true) {

    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> states;
    std::vector<ActionDist> distributions;
    std::vector<Player> players;
    
    std::vector<Value> outcomes;

    std::vector<SymmetryIdx> allSymmetries;  // Contains all symmetries if symmetrizer exists.

    if (symmetrizer != nullptr) {
        for (int i = 0; i < symmetrizer->numSymmetries(); ++i) {
            allSymmetries.push_back(i);
        }
    }

    // Initialize the UCT tree.
    UCTTree<ImplNode, State, ACTION_SIZE> tree {
        std::move(rootNode),
        dirEps,
        dirAlpha,
        initQMethod,
        dropParent,
        symmetrizer,
        addNoise,
    };

    int moveCount = 0;

    while (!tree.getDecisionNode()->isTerminal()) {

        // with some probability, decide to make a fast play!
        bool doFullSearch = GetRandom().UniformInt(0, 3) == 0;

        if (doFullSearch) {
            if (symmetrizer != nullptr) {
                // Symmetrize the state and add to data.
                std::vector<State> symmetrizedStates = symmetrizer->symmetrizeState(
                    tree.getDecisionNode()->getGameState(), allSymmetries);

                states.reserve(states.size() + symmetrizedStates.size());
                states.insert(states.end(), symmetrizedStates.begin(), symmetrizedStates.end());

            } else {
                states.push_back(tree.getDecisionNode()->getGameState());
            }
        }

        // Perform `numTraversals` many search iterations.
        int traversals = 0;
        int numTraversalsOnThisTurn = doFullSearch ? numTraversals : numTraversals / 6; // In the paper, (600, 100). 
        
        while (traversals < numTraversalsOnThisTurn) {
            auto [leaves, trav] = tree.searchAndGetLeaves(maxBatchSize, maxQueueSize, network, U_WEIGHT);

            if (leaves.size() > 0) {
                tree.evaluateAndBackpropLeaves(leaves, network);
            }

            traversals += trav;
        }

        // Generate a PDF from the visit counts.
        ActionDist visits = tree.getDecisionNode()->getEdgeStatistics()->m_numVisits; // THIS LINE 
        ActionDist pdf = visits / visits.sum();

        // Raise it to 0.98f (temp ~ 1) if early game, else 10.0f (temp -> 0), then renormalize.
        if (moveCount < EARLY_GAME_CUTOFF) {
            pdf = pdf.pow(EARLY_GAME_EXP);
        } else {
            pdf = pdf.pow(REST_GAME_EXP);
        }

        pdf = pdf / pdf.sum();

        // Generate a CDF from the PDF.
        ActionDist cdf = pdf.cumsum();
        cdf = cdf / cdf[ACTION_SIZE - 1];

        if (doFullSearch) {
            if (symmetrizer != nullptr) {
                // Symmetrize the distributions and add to data.
                std::vector<ActionDist> symmetrizedDists = symmetrizer->symmetrizeActionDist(
                    pdf, allSymmetries);

                distributions.reserve(distributions.size() + symmetrizedDists.size());
                distributions.insert(distributions.end(), symmetrizedDists.begin(), symmetrizedDists.end());

            } else {
                distributions.push_back(pdf);
            }
        }

        // Sample from the CDF.
        int action = GetRandom().SampleCDF(std::vector<float>(cdf.begin(), cdf.end()));

        if (doFullSearch) {
            // Record the player that just took the action.
            players.push_back(tree.getDecisionNode()->getPlayer());
        }

        // Play the action by rerooting the tree and updating the state.
        tree.advanceDecision(action);

        ++moveCount;
    }

    std::array<Value, 2> rewards = tree.getDecisionNode()->getRewards();

    outcomes.reserve(states.size());

    for (Player player : players) {
        switch (player) {
        case Player::ZERO:
            if (symmetrizer != nullptr) {
                for (int i = 0; i < symmetrizer->numSymmetries(); ++i) {
                    outcomes.push_back(rewards[0]);
                }
            } else {
                outcomes.push_back(rewards[0]);
            }
            
            break;

        case Player::ONE:
            if (symmetrizer != nullptr) {
                for (int i = 0; i < symmetrizer->numSymmetries(); ++i) {
                    outcomes.push_back(rewards[1]);
                }
            } else {
                outcomes.push_back(rewards[1]);
            }

            break;

        default:
            assert(false);

            if (symmetrizer != nullptr) {
                for (int i = 0; i < symmetrizer->numSymmetries(); ++i) {
                    outcomes.push_back(0.0f);
                }
            } else {
                outcomes.push_back(0.0f);
            }
        }
    }

    return { states, distributions, outcomes };
}

/**
 * Runs `numGames` many games of self-play and collates all the data.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @note See `selfPlay()` for more details.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
std::tuple<std::vector<State>, std::vector<GameActionDist<ACTION_SIZE>>, std::vector<Value>>
runIteration(INetwork<State, ACTION_SIZE>* network, int numGames,
             int numTraversals, int maxBatchSize, int maxQueueSize,
             float dirEps, float dirAlpha, InitQ initQMethod, bool dropParent,
             ISymmetrizer<State, ACTION_SIZE>* symmetrizer, bool addNoise = true) {

    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> allStates;
    std::vector<ActionDist> allDistributions;
    std::vector<Value> allOutcomes;

    for (int t = 0; t < numGames; ++t) {
        std::unique_ptr<GameNode<ImplNode, State, ACTION_SIZE>> rootNode = std::make_unique<ImplNode>();

        auto [states, distributions, outcomes] = selfPlay(
            std::move(rootNode),
            network,
            numTraversals,
            maxBatchSize,
            maxQueueSize,
            dirEps,
            dirAlpha,
            initQMethod,
            dropParent,
            symmetrizer,
            addNoise
        );

        allStates.reserve(allStates.size() + states.size());
        allStates.insert(allStates.end(), states.begin(), states.end());

        allDistributions.reserve(allDistributions.size() + distributions.size());
        allDistributions.insert(allDistributions.end(), distributions.begin(), distributions.end());

        allOutcomes.reserve(allOutcomes.size() + outcomes.size());
        allOutcomes.insert(allOutcomes.end(), outcomes.begin(), outcomes.end());

        std::cout << t + 1 << " games played, " << allStates.size() << " states collected.\n";
    }

    assert(allStates.size() == allDistributions.size());
    assert(allStates.size() == allOutcomes.size());

    return { allStates, allDistributions, allOutcomes };
}

} // namespace SPRL

#endif